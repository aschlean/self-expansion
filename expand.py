from typing import List, Literal
from db import driver
import structured_gen as sg
from pydantic import BaseModel, Field, field_validator
from rich import print


# Define the data structures
class Question(BaseModel):
    type: Literal["Question"]
    text: str


class Concept(BaseModel):
    type: Literal["Concept"]
    # Text must be lowercase
    text: str = Field(pattern=r"^[a-z ]+$")


class ConceptWithLinks(Concept):
    relationship_type: Literal["IS_A", "AFFECTS", "CONNECTS_TO"]


class Answer(BaseModel):
    type: Literal["Answer"]
    text: str


class Feature(BaseModel):
    type: Literal["Feature"]
    text: str


class PartnerFeature(BaseModel):
    type: Literal["PartnerFeature"]
    text: str


class Usecase(BaseModel):
    type: Literal["Usecase"]
    text: str


class UserGroup(BaseModel):
    type: Literal["UserGroup"]
    text: str


# Permitted response formats
class FromQuestion(BaseModel):
    """If at a question, may generate an answer."""

    answer: List[Answer]


class FromConcept(BaseModel):
    """If at a concept, may produce questions or relate to concepts"""

    questions: List[Question]
    concepts: List[ConceptWithLinks]


class FromAnswer(BaseModel):
    """If at an answer, may generate concepts or new questions"""

    concepts: List[Concept]
    questions: List[Question]


class FromCore(BaseModel):
    """From core node, can ONLY connect to existing features and partner features"""
    # These should only be for connecting, not creating new ones
    features: List[str]  # Changed from List[Feature] to List[str] - just the text to connect to
    partner_features: List[str]  # Changed from List[PartnerFeature] to List[str]


class FromFeature(BaseModel):
    """From feature node, can generate usecases by combining with partner features"""
    usecases: List[Usecase] = Field(description="Generate usecases that specifically combine this feature with a partner feature. Each usecase must clearly indicate which partner feature it uses.")
    partner_features: List[str]  # Only for connecting to existing ones


class FromPartnerFeature(BaseModel):
    """From partner feature node, can generate usecases by combining with features"""
    features: List[str]  # Only for connecting to existing ones


class FromUsecase(BaseModel):
    """From usecase node, can connect to user groups. Must generate at least one user group."""
    user_groups: List[UserGroup]
    
    # Add validator to ensure at least one user group is generated
    @field_validator('user_groups')
    def validate_user_groups(cls, v):
        if not v:
            raise ValueError("Must generate at least one user group for each usecase")
        return v


class FromUserGroup(BaseModel):
    """From user group node, can connect to usecases and features"""
    usecases: List[Usecase] = Field(description="""
        Generate new usecases specific to this user group.
        IMPORTANT: 
        - Do not recreate similar usecases that already exist
        - Focus on different feature combinations than what's already been used
        - Each usecase must solve a distinct problem for this user group
    """)
    features: List[Feature]


# Create a core node if it doesn't exist, or return the existing core node ID
def get_or_make_core(question: str):
    with driver.session() as session:
        # Check if node exists and get its ID
        result = session.run(
            """
            MATCH (n:Core {text: $question})
            RETURN n.id as id
        """,
            question=question,
        )

        data = result.data()
        if len(data) > 0:
            return data[0]["id"]

        # Create new node with UUID if it doesn't exist
        result = session.run(
            """
            MERGE (n:Core {text: $question})
            ON CREATE SET n.id = randomUUID()
            RETURN n.id as id
        """,
            question=question,
        )
        data = result.data()
        if len(data) > 0:
            return data[0]["id"]
        else:
            raise ValueError(f"Failed to create new core node for question: {question}")


def load_neighbors(node_id: str, distance: int = 1):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (node {id: $node_id})-[rel]-(neighbor)
            WHERE type(rel) <> "TRAVERSED"
            RETURN
                node.id as node_id,
                node.text as node_text,
                type(rel) as rel_type,
                neighbor.id as neighbor_id,
                neighbor.text as neighbor_text,
                labels(neighbor)[0] as neighbor_type,
                labels(node)[0] as node_type
        """,
            node_id=node_id,
        )
        return result.data()


def load_node(node_id: str):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (node {id: $node_id})
            RETURN node.id as node_id, node.text as node_text, labels(node)[0] as label
        """,
            node_id=node_id,
        )
        return result.single(strict=True)


# Node linking functions
# Relationship Types (all have curiosity score 0-1):
#   RAISES -> (Concept/Core to Question)
#   ANSWERS -> (Answer to Question)
#   SUGGESTS -> (Answer to Concept)
#   RELATES_TO -> (Concept to Concept)
def question_to_concept(question: str, concept: str):
    try:
        question_embedding = sg.embed(question)
        concept_embedding = sg.embed(concept)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (concept:Concept {text: $concept})
            ON CREATE SET concept.id = randomUUID(), concept.embedding = $concept_embedding
            MERGE (concept)-[:RAISES]->(question)
        """,
            question=question,
            concept=concept,
            question_embedding=question_embedding,
            concept_embedding=concept_embedding,
        )


def question_to_answer(question: str, answer: str):
    try:
        question_embedding = sg.embed(question)
        answer_embedding = sg.embed(answer)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (answer:Answer {text: $answer})
            ON CREATE SET answer.id = randomUUID(), answer.embedding = $answer_embedding
            MERGE (answer)-[:ANSWERS]->(question)
        """,
            question=question,
            answer=answer,
            question_embedding=question_embedding,
            answer_embedding=answer_embedding,
        )


def concept_to_concept(
    concept1: str,
    concept2: str,
    relationship_type: Literal["IS_A", "AFFECTS", "CONNECTS_TO"],
):
    # If the concepts are the same, don't create a relationship
    if concept1 == concept2:
        return

    try:
        concept1_embedding = sg.embed(concept1)
        concept2_embedding = sg.embed(concept2)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        query = f"""
            MERGE (concept1:Concept {{text: $concept1}})
            ON CREATE SET concept1.id = randomUUID(), concept1.embedding = $concept1_embedding
            MERGE (concept2:Concept {{text: $concept2}})
            ON CREATE SET concept2.id = randomUUID(), concept2.embedding = $concept2_embedding
            MERGE (concept1)-[:{relationship_type}]->(concept2)
        """
        session.run(
            query,
            concept1=concept1,
            concept2=concept2,
            concept1_embedding=concept1_embedding,
            concept2_embedding=concept2_embedding,
        )


def concept_to_question(concept: str, question: str):
    try:
        concept_embedding = sg.embed(concept)
        question_embedding = sg.embed(question)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (concept:Concept {text: $concept})
            ON CREATE SET concept.id = randomUUID(), concept.embedding = $concept_embedding
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (concept)-[:RAISES]->(question)
        """,
            concept=concept,
            question=question,
            concept_embedding=concept_embedding,
            question_embedding=question_embedding,
        )


# Core-specific functions
def core_to_question(core: str, question: str):
    try:
        core_embedding = sg.embed(core)
        question_embedding = sg.embed(question)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (core:Core {text: $core})
            ON CREATE SET core.id = randomUUID(), core.embedding = $core_embedding
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (core)-[:RAISES]->(question)
        """,
            core=core,
            question=question,
            core_embedding=core_embedding,
            question_embedding=question_embedding,
        )


def concept_to_core(concept: str, core: str):
    try:
        concept_embedding = sg.embed(concept)
        core_embedding = sg.embed(core)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (concept:Concept {text: $concept})
            ON CREATE SET concept.id = randomUUID(), concept.embedding = $concept_embedding
            MERGE (core:Core {text: $core})
            ON CREATE SET core.id = randomUUID(), core.embedding = $core_embedding
            MERGE (concept)-[:EXPLAINS]->(core)
        """,
            concept=concept,
            core=core,
            concept_embedding=concept_embedding,
            core_embedding=core_embedding,
        )


def answer_to_concept(answer: str, concept: str):
    try:
        answer_embedding = sg.embed(answer)
        concept_embedding = sg.embed(concept)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (answer:Answer {text: $answer})
            ON CREATE SET answer.id = randomUUID(), answer.embedding = $answer_embedding
            MERGE (concept:Concept {text: $concept})
            ON CREATE SET concept.id = randomUUID(), concept.embedding = $concept_embedding
            MERGE (answer)-[:SUGGESTS]->(concept)
        """,
            answer=answer,
            concept=concept,
            answer_embedding=answer_embedding,
            concept_embedding=concept_embedding,
        )


def answer_to_question(answer: str, question: str):
    try:
        answer_embedding = sg.embed(answer)
        question_embedding = sg.embed(question)
    except Exception as e:
        print(f"Skipping connection due to embedding error: {e}")
        return

    with driver.session() as session:
        session.run(
            """
            MERGE (answer:Answer {text: $answer})
            ON CREATE SET answer.id = randomUUID(), answer.embedding = $answer_embedding
            MERGE (question:Question {text: $question})
            ON CREATE SET question.id = randomUUID(), question.embedding = $question_embedding
            MERGE (answer)-[:ANSWERS]->(question)
        """,
            answer=answer,
            question=question,
            answer_embedding=answer_embedding,
            question_embedding=question_embedding,
        )


def record_traversal(
    from_node_id: str,
    to_node_id: str,
    traversal_type: Literal["random", "core", "neighbor"],
):
    with driver.session() as session:
        session.run(
            """
            MERGE (from_node {id: $from_node_id})
            MERGE (to_node {id: $to_node_id})
            MERGE (from_node)-[:TRAVERSED {timestamp: timestamp(), traversal_type: $traversal_type}]->(to_node)
        """,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            traversal_type=traversal_type,
        )


def clear_db():
    with driver.session() as session:
        session.run(
            """
            MATCH (n) DETACH DELETE n
        """
        )


def random_node_id():
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n) RETURN n.id as id LIMIT 1
        """
        )
        return result.single(strict=True)["id"]


def format_node_neighborhood(node_id, truncate: bool = True):
    # Create ID mapping using ASCII uppercase letters (AA, AB, AC, etc.)
    id_counter = 0
    uuid_to_simple_mapping = {}
    simple_to_uuid_mapping = {}

    def get_simple_id():
        nonlocal id_counter
        # Generate IDs like AA, AB, ..., ZZ
        first = chr(65 + (id_counter // 26))
        second = chr(65 + (id_counter % 26))
        id_counter += 1
        return f"NODE-{first}{second}"

    node = load_node(node_id)
    neighbors = load_neighbors(node_id)
    neighbors_string = f"{node['label'].upper()} {node['node_text']}\n"

    # Add direct neighbors
    if len(neighbors) > 0:
        neighbors_string += "\nDIRECT CONNECTIONS:\n"
        for neighbor in neighbors:
            text = neighbor["neighbor_text"]
            if truncate:
                text = text[:70] + "..." if len(text) > 70 else text
            simple_id = get_simple_id()
            simple_to_uuid_mapping[simple_id] = neighbor["neighbor_id"]
            uuid_to_simple_mapping[neighbor["neighbor_id"]] = simple_id
            neighbors_string += f"{simple_id:<8} {neighbor['rel_type']:<12} {neighbor['neighbor_type'].upper():<10} {text}\n"

    # Add semantically related nodes
    related = find_related_nodes(node_id)

    if len(related) > 0:
        neighbors_string += "\nSEMANTICALLY RELATED:\n"
        for node_type, nodes in related.items():
            if nodes:  # Only add section if there are related nodes
                neighbors_string += f"\n{node_type}s:\n"
                for n in nodes:
                    text = n["node_text"]
                    if truncate:
                        text = text[:70] + "..." if len(text) > 70 else text
                    simple_id = get_simple_id()
                    simple_to_uuid_mapping[simple_id] = n["node_id"]
                    uuid_to_simple_mapping[n["node_id"]] = simple_id
                    neighbors_string += f"{simple_id:<8} {n['score']:<12.2f} {node_type.upper():<10} {text}\n"

    return neighbors_string, uuid_to_simple_mapping, simple_to_uuid_mapping


def find_related_nodes(node_id: str):
    with driver.session() as session:
        result = {}
        # Updated to include all our new node types
        for node_type in ["Feature", "PartnerFeature", "Usecase", "UserGroup"]:
            result[node_type] = session.run(
                """
                MATCH (m {id: $node_id})
                WHERE m.embedding IS NOT NULL
                CALL db.index.vector.queryNodes(
                    $vector_index_name,
                    $limit,
                    m.embedding
                )
                YIELD node, score
                RETURN node.id as node_id, node.text as node_text, score
            """,
                node_id=node_id,
                vector_index_name=f"{node_type.lower()}_embedding",
                limit=10,
            ).data()
        return result


def remove_index(index_name: str):
    with driver.session() as session:
        session.run(
            f"""
            DROP INDEX {index_name} IF EXISTS
        """
        )


def main(
    do_clear_db=True, 
    purpose="Find high quality use cases for Firecrawl with Outlines"
):
    # Define our core features
    firecrawl_features = [
        """Convert any webpage into clean, structured markdown via /scrape endpoint.
        Input: Any URL
        Output: Clean markdown text with preserved structure, links, and important content
        Use for: Converting web content into a format suitable for LLM processing""",
        
        """Extract webpage metadata via /scrape endpoint.
        Input: Any URL
        Output: Title, description, OpenGraph tags, author info
        Use for: Getting structured metadata about web pages""",
        
        """Batch process multiple URLs via /scrape endpoint.
        Input: List of URLs
        Output: Markdown and metadata for each URL
        Use for: Processing entire content collections at once""",

        """Discover and process all pages on a website via /crawl endpoint.
        Input: Starting URL
        Output: Processed content from all discovered pages
        Use for: Complete website content extraction""",
        
        """Generate website structure map via /map endpoint.
        Input: Website URL
        Output: Complete site structure and URL list
        Use for: Content auditing and site mapping""",

        """Extract specific data from web content via /extract endpoint.
        Input: URL + extraction schema
        Output: Structured data matching schema
        Use for: Converting web content into structured data""",
        
        """Extract unstructured data via /extract endpoint.
        Input: URL + extraction prompt
        Output: Extracted information based on a prompt
        Use for: Flexible data extraction without rigid schemas"""
    ]

    outlines_features = [
        """Generate text from LLM matching exact JSON schema.
        Input: Schema + prompt
        Output: Text conforming to schema
        Use for: Getting structured text responses""",
        
        """Generate text from LLM following grammar (CFG)rules.
        Input: Grammar definition + prompt
        Output: Text following grammar
        Use for: Getting consistently formatted text""",
        
        """Generate text from LLM with type constraints.
        Input: Pydantic model + prompt
        Output: Type-safe text data
        Use for: Getting properly typed data from LLMs""",
        
        """Stream generation results from LLM in real-time.
        Input: Generation request
        Output: Token stream
        Use for: Real-time text processing""",
        
        """Validate outputs from LLM against schemas.
        Input: Generated text + schema
        Output: Validation result
        Use for: Ensuring output quality"""
    ]

    # Clear the database if requested
    if do_clear_db:
        print("WARNING: Clearing the database")
        clear_db()

    # Load our predefined features
    load_initial_data(
        features=firecrawl_features,
        partner_features=outlines_features,
        core_text=purpose
    )

    # Create the core node and get its ID
    current_node_id = get_or_make_core(purpose)
    core_node_id = current_node_id

    # Get embedding dimensions
    embedding_dimensions = len(sg.embed(purpose))

    # Remove existing indices
    remove_index("core_id")
    remove_index("feature_embedding")
    remove_index("partner_feature_embedding")
    remove_index("usecase_embedding")
    remove_index("user_group_embedding")

    # Create indices
    with driver.session() as session:
        # Create regular indices
        index_queries = [
            "CREATE INDEX core_id IF NOT EXISTS FOR (n:Core) ON (n.id)",
            "CREATE INDEX feature_id IF NOT EXISTS FOR (n:Feature) ON (n.id)",
            "CREATE INDEX partner_feature_id IF NOT EXISTS FOR (n:PartnerFeature) ON (n.id)",
            "CREATE INDEX usecase_id IF NOT EXISTS FOR (n:Usecase) ON (n.id)",
            "CREATE INDEX user_group_id IF NOT EXISTS FOR (n:UserGroup) ON (n.id)",
        ]

        # Create vector indices
        vector_index_queries = []
        for node_type in ["Feature", "PartnerFeature", "Usecase", "UserGroup"]:
            vector_index_queries.append(
                f"""
                CREATE VECTOR INDEX {node_type.lower()}_embedding IF NOT EXISTS
                FOR (n:{node_type}) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $embedding_dimensions,
                        `vector.similarity_function`: 'COSINE'
                    }}
                }}
            """
            )

        # Execute all queries
        for query in index_queries + vector_index_queries:
            session.run(query, embedding_dimensions=embedding_dimensions)

    # Loop through the main code
    history = []
    while True:
        # Get current node
        current_node = load_node(current_node_id)
        current_node_text = current_node["node_text"]
        current_node_label = current_node["label"]

        # Get the user prompt
        prompt = (
            "\n".join([f"{n['label'].upper()} {n['node_text']}" for n in history])
            + f"\nCurrent node: {current_node_label.upper()} {current_node_text}"
        )
        prompt = "Here is the traversal history:\n" + prompt
        # prompt += f"Here are nodes related to the current node:\n" +\
        #       format_node_neighborhood(current_node_id, truncate=False)

        # Check current node type and set result format
        result_format = None
        if current_node_label == "Core":
            result_format = FromCore
        elif current_node_label == "Feature":
            result_format = FromFeature
        elif current_node_label == "PartnerFeature":
            result_format = FromPartnerFeature
        elif current_node_label == "Usecase":
            result_format = FromUsecase
        elif current_node_label == "UserGroup":
            result_format = FromUserGroup
        else:
            raise ValueError(f"Unknown node type: {current_node_label}")

        # Get the system prompt
        system_prompt = f"""
        You are a superintelligent AI building a self-expanding knowledge graph.
        Your goal is to achieve the core directive "{purpose}".

        Generate an expansion of the current node. Based on the node type, an expansion may include:

        1. At CORE node:
           - ONLY connect to existing features and partner features
           - DO NOT generate new features or partner features
           - DO NOT create any other node types

        2. At FEATURE node:
           - EXAMINE connected partner features in the graph
           - GENERATE use cases that specifically combine:
             * The current feature AND
             * One connected partner feature
           - Each usecase must clearly describe how the feature and partner feature work together
           - DO NOT generate new features or partner features

        3. At PARTNER_FEATURE node:
           - EXAMINE connected features in the graph
           - GENERATE use cases that specifically combine:
             * The current partner feature AND
             * One connected feature
           - Each usecase must clearly describe how they work together
           - DO NOT generate new partner features

        4. At USECASE node:
           - ONLY generate user groups that would benefit from this use case
           - MUST generate at least one specific user group for EACH use case
           - Each user group must be specific and well-defined
           - DO NOT generate any other node types

        5. At USER_GROUP node:
           - GENERATE use cases specific to this user group
           - Each usecase must be DISTINCTLY DIFFERENT from existing ones
           - Use different feature + partner feature combinations
           - Focus on unique problems and solutions
           - DO NOT recreate similar usecases with minor variations
           - CONNECT to existing features the group would use
           - DO NOT generate new features

        IMPORTANT:
        - Each use case MUST combine one feature with one partner feature
        - Always specify the feature + partner feature combination in use case descriptions
        - ONLY use features and partner features from the initial lists
        - DO NOT generate new features or partner features at any point
        """

        # Generate an expansion
        try:
            # print("generating expansion")
            # result = sg.generate(
            #     sg.messages(user=prompt, system=system_prompt),
            #     response_format=result_format
            # )

            print(prompt)

            result = sg.generate_by_schema(
                sg.messages(user=prompt, system=system_prompt),
                result_format.model_json_schema(),
            )
            expansion = result_format.model_validate_json(
                result.choices[0].message.content
            )
        except Exception as e:
            print(f"Error generating expansion: {e}")
            current_node_id = core_node_id
            continue

        # Link the new nodes based on current node type
        if current_node_label == "Core":
            for feature in expansion.features:
                core_has_feature(current_node_text, feature)
            for partner_feature in expansion.partner_features:
                core_connects_partner_feature(current_node_text, partner_feature)

        elif current_node_label == "Feature":
            for usecase in expansion.usecases:
                feature_enables_usecase(current_node_text, usecase.text)
            for partner_feature in expansion.partner_features:
                partner_feature_connects_feature(partner_feature, current_node_text)

        elif current_node_label == "PartnerFeature":
            for feature in expansion.features:
                partner_feature_connects_feature(current_node_text, feature)

        elif current_node_label == "Usecase":
            for user_group in expansion.user_groups:
                usecase_benefits_user_group(current_node_text, user_group.text)

        elif current_node_label == "UserGroup":
            for usecase in expansion.usecases:
                user_group_uses_usecase(current_node_text, usecase.text)
            for feature in expansion.features:
                user_group_uses_feature(current_node_text, feature.text)

        # Grab the current node's neighbors and format them for display
        neighbors = load_neighbors(current_node_id)

        # Formatting the neighbor table
        (
            neighbors_string,
            uuid_to_simple_mapping,
            simple_to_uuid_mapping,
        ) = format_node_neighborhood(current_node_id)

        # Check graph balance and usecase count
        balance_message, usecase_count = check_graph_balance()
        if usecase_count >= 30:  # Stop if we've reached 30 usecases
            print("\nReached target number of use cases (30). Stopping expansion.")
            break

        # Choose a new node if there are any neighbors
        if len(neighbors) > 0:
            old_node_id = current_node_id

            print(
                "----------------------------------------------------------------------------------"
            )
            print(neighbors_string)

            # Construct selectable nodes
            selectable_nodes = set()
            for neighbor in neighbors:
                # Add the neighbor's simple ID
                selectable_nodes.add(uuid_to_simple_mapping[neighbor["neighbor_id"]])

            # Add all the keys in the uuid_to_simple_mapping
            selectable_nodes.update(simple_to_uuid_mapping.keys())

            selectable_nodes.add("random")
            # selectable_nodes.add('core')

            # Remove the current node from the selectable nodes if it's in there
            # This prevents the AI from choosing the current node again.
            if current_node_id in selectable_nodes:
                selectable_nodes.remove(current_node_id)

            choice_prompt = (
                prompt
                + "Select a node to traverse to. Respond with the node ID."
                + balance_message
                + "You will generate a new expansion of the node you traverse to."
                + "You will not be able to choose the current node."
                + "You may choose 'random' to choose a random node."
            )
            # "You may also choose 'core' to return to the core node, " + \
            # "or 'random' to choose a random node."

            node_selection = sg.choose(
                sg.messages(user=choice_prompt, system=system_prompt),
                choices=list(selectable_nodes),
            )

            is_random = node_selection == "random"
            is_core = node_selection == "core"

            if is_random:
                current_node_id = random_node_id()
            elif is_core:
                current_node_id = core_node_id
            else:
                current_node_id = simple_to_uuid_mapping[node_selection]

            # Print the node label + text
            print(f"SELECTED {node_selection} {current_node_id}")
            node = load_node(current_node_id)
            print(f"SELECTED {node['label'].upper()} {node['node_text']}\n")

            history.append(current_node)

            traversal_type = (
                "random" if is_random else "core" if is_core else "neighbor"
            )
            record_traversal(old_node_id, current_node_id, traversal_type)


def core_has_feature(core: str, feature: str):
    with driver.session() as session:
        # First verify the feature exists
        result = session.run(
            """
            MATCH (feature:Feature {text: $feature})
            RETURN feature
            """,
            feature=feature
        )
        if not result.data():
            print(f"Skipping connection - feature does not exist: {feature}")
            return

        # Then create the connection
        session.run(
            """
            MERGE (core:Core {text: $core})
            ON CREATE SET core.id = randomUUID()
            WITH core
            MATCH (feature:Feature {text: $feature})
            MERGE (core)-[:HAS]->(feature)
            """,
            core=core,
            feature=feature
        )


def core_connects_partner_feature(core: str, partner_feature: str):
    with driver.session() as session:
        # First verify the partner feature exists
        result = session.run(
            """
            MATCH (partner:PartnerFeature {text: $partner_feature})
            RETURN partner
            """,
            partner_feature=partner_feature
        )
        if not result.data():
            print(f"Skipping connection - partner feature does not exist: {partner_feature}")
            return

        # Then create the connection
        session.run(
            """
            MERGE (core:Core {text: $core})
            ON CREATE SET core.id = randomUUID()
            WITH core
            MATCH (partner:PartnerFeature {text: $partner_feature})
            MERGE (core)-[:CONNECTS]->(partner)
            """,
            core=core,
            partner_feature=partner_feature
        )


def partner_feature_connects_feature(partner_feature: str, feature: str):
    with driver.session() as session:
        # First verify both nodes exist
        result = session.run(
            """
            MATCH (partner:PartnerFeature {text: $partner_feature})
            MATCH (feature:Feature {text: $feature})
            RETURN partner, feature
            """,
            partner_feature=partner_feature,
            feature=feature
        )
        if not result.data():
            print(f"Skipping connection - one or both nodes don't exist: {partner_feature} -> {feature}")
            return

        # Then create only the connection
        session.run(
            """
            MATCH (partner:PartnerFeature {text: $partner_feature})
            MATCH (feature:Feature {text: $feature})
            MERGE (partner)-[:CONNECTS]->(feature)
            """,
            partner_feature=partner_feature,
            feature=feature
        )


def feature_enables_usecase(feature: str, usecase: str):
    with driver.session() as session:
        # First verify the feature exists
        result = session.run(
            """
            MATCH (feature:Feature {text: $feature})
            RETURN feature
            """,
            feature=feature
        )
        if not result.data():
            print(f"Skipping connection - feature does not exist: {feature}")
            return

        # Create usecase and connection
        try:
            usecase_embedding = sg.embed(usecase)
            session.run(
                """
                MATCH (feature:Feature {text: $feature})
                MERGE (usecase:Usecase {text: $usecase})
                ON CREATE SET usecase.id = randomUUID(), 
                             usecase.embedding = $usecase_embedding
                MERGE (feature)-[:ENABLES]->(usecase)
                """,
                feature=feature,
                usecase=usecase,
                usecase_embedding=usecase_embedding
            )
        except Exception as e:
            print(f"Error creating usecase connection: {e}")


def usecase_benefits_user_group(usecase: str, user_group: str):
    with driver.session() as session:
        # First verify the usecase exists
        result = session.run(
            """
            MATCH (usecase:Usecase {text: $usecase})
            RETURN usecase
            """,
            usecase=usecase
        )
        if not result.data():
            print(f"Skipping connection - usecase does not exist: {usecase}")
            return

        # Create user group (this is allowed) and connection
        try:
            user_group_embedding = sg.embed(user_group)
            session.run(
                """
                MATCH (usecase:Usecase {text: $usecase})
                MERGE (user_group:UserGroup {text: $user_group})
                ON CREATE SET user_group.id = randomUUID(),
                             user_group.embedding = $user_group_embedding
                MERGE (usecase)-[:BENEFITS]->(user_group)
                """,
                usecase=usecase,
                user_group=user_group,
                user_group_embedding=user_group_embedding
            )
        except Exception as e:
            print(f"Error creating user group connection: {e}")


def user_group_uses_usecase(user_group: str, usecase: str):
    with driver.session() as session:
        # First verify the user group exists
        result = session.run(
            """
            MATCH (user_group:UserGroup {text: $user_group})
            RETURN user_group
            """,
            user_group=user_group
        )
        if not result.data():
            print(f"Skipping connection - user group does not exist: {user_group}")
            return

        # Create usecase (this is allowed) and connection
        try:
            usecase_embedding = sg.embed(usecase)
            session.run(
                """
                MATCH (user_group:UserGroup {text: $user_group})
                MERGE (usecase:Usecase {text: $usecase})
                ON CREATE SET usecase.id = randomUUID(),
                             usecase.embedding = $usecase_embedding
                MERGE (user_group)-[:USES]->(usecase)
                """,
                user_group=user_group,
                usecase=usecase,
                usecase_embedding=usecase_embedding
            )
        except Exception as e:
            print(f"Error creating usecase connection: {e}")


def user_group_uses_feature(user_group: str, feature: str):
    with driver.session() as session:
        # First verify both nodes exist
        result = session.run(
            """
            MATCH (user_group:UserGroup {text: $user_group})
            MATCH (feature:Feature {text: $feature})
            RETURN user_group, feature
            """,
            user_group=user_group,
            feature=feature
        )
        if not result.data():
            print(f"Skipping connection - one or both nodes don't exist: {user_group} -> {feature}")
            return

        # Then create only the connection
        session.run(
            """
            MATCH (user_group:UserGroup {text: $user_group})
            MATCH (feature:Feature {text: $feature})
            MERGE (user_group)-[:USES]->(feature)
            """,
            user_group=user_group,
            feature=feature
        )


def load_initial_data(features: List[str], partner_features: List[str], core_text: str):
    """
    Load initial features and partner features into the graph and connect them to the core node.
    
    Args:
        features: List of feature descriptions
        partner_features: List of partner feature descriptions
        core_text: The core node text/purpose
    """
    print("Loading initial data...")
    
    # First create/get the core node
    core_id = get_or_make_core(core_text)
    
    with driver.session() as session:
        # First create all features
        for feature in features:
            try:
                print(f"Creating feature: {feature}")
                feature_embedding = sg.embed(feature)
                session.run(
                    """
                    MERGE (feature:Feature {text: $feature})
                    ON CREATE SET feature.id = randomUUID(), 
                                 feature.embedding = $feature_embedding
                    """,
                    feature=feature,
                    feature_embedding=feature_embedding
                )
            except Exception as e:
                print(f"Error creating feature {feature}: {e}")
                
        # Then create all partner features
        for partner_feature in partner_features:
            try:
                print(f"Creating partner feature: {partner_feature}")
                partner_embedding = sg.embed(partner_feature)
                session.run(
                    """
                    MERGE (partner:PartnerFeature {text: $partner_feature})
                    ON CREATE SET partner.id = randomUUID(),
                                 partner.embedding = $partner_embedding
                    """,
                    partner_feature=partner_feature,
                    partner_embedding=partner_embedding
                )
            except Exception as e:
                print(f"Error creating partner feature {partner_feature}: {e}")
        
        # Now connect everything to core
        for feature in features:
            try:
                print(f"Connecting feature to core: {feature}")
                core_has_feature(core_text, feature)
            except Exception as e:
                print(f"Error connecting feature {feature}: {e}")
                
        for partner_feature in partner_features:
            try:
                print(f"Connecting partner feature to core: {partner_feature}")
                core_connects_partner_feature(core_text, partner_feature)
            except Exception as e:
                print(f"Error connecting partner feature {partner_feature}: {e}")
                
    print("Initial data loading complete!")


def check_graph_balance():
    with driver.session() as session:
        # First check if we have any usecases
        result = session.run("""
            MATCH (u:Usecase)
            RETURN count(DISTINCT u) as usecase_count
        """)
        usecase_data = result.single()
        if not usecase_data or usecase_data["usecase_count"] == 0:
            return "", 0  # Return both message and count when no usecases exist
            
        # Then check user groups
        result = session.run("""
            MATCH (g:UserGroup)
            RETURN count(DISTINCT g) as group_count
        """)
        group_data = result.single()
        group_count = group_data["group_count"] if group_data else 0
        
        usecase_count = usecase_data["usecase_count"]
        if usecase_count - group_count >= 2:
            print("Consider traversing to a usecase node to generate more user groups.")
            return "Consider traversing to a usecase node to generate more user groups.", usecase_count
        return "", usecase_count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a self-expanding knowledge graph around Firecrawl and Outlines integration."
    )

    parser.add_argument(
        "--do-clear-db",
        "--do_clear_db",
        action="store_true",
        help="If set, clear the database before proceeding.",
    )

    args = parser.parse_args()

    main(args.do_clear_db)
