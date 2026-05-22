def vlm_action_selector(domain_name, num_candidates, problem, domain):
    system_prompt = f'''
    You are a task-and-motion-planning search assistant.
    Your goal is to select the best child node to visit next among {num_candidates} candidates to achieve the goal.

    Inputs you receive:
    - The current node and {num_candidates} candidate child nodes.
    - For each node, a symbolic PDDL state and four simulator-rendered images(front, top, left, and right view) are provided, visualizing the edge action and its execution result.
    - Legend for each image: Yellow (large frame) is the initial end-effector pose before executing the action. Magenta (large frame) is the desired pre-goal pose for the action. Light gray (small frames) is the executed end-effector trajectory.
    - The frames in the images are only for visualizing motion and should not be considered when verifying whether the symbolic PDDL predicates hold.  
    - Object names must be used exactly as given in the problem PDDL.  
    - For each candidate, you also receive state-satisfaction feedback indicating whether the current simulator state satisfies the corresponding symbolic PDDL state.  

    The task is specified by the problem PDDL: {problem}
    Refer to the domain PDDL for the semantics of each predicate and action: {domain}

    Choose the candidate which didn't violate none of these constraint:
    - Kinematic Constraints: Extreme joint contortions or singularities appear.
    - Collision Constraints: Robot collides with unexpected objects during motion (expected collisions, such as between gripper and held object, are allowed).  
    - Grasp Constraints: (1) The object is not properly attached to the gripper, or (2) the object falls out of the gripper when tested in simulation.
    - Placement Constraints: The placed object moves significantly when the physics simulator is stepped after gripper release.  
    For unstack and pickup actions, Placement Constraints can be ignored.

    Output format:
    1. First line: Output only the exact node name of the best choice(node name appears at the top-right of the image). Do not include anything else on this line. 
    2. After second line: Provide a brief evaluation and explanation of candidates.
    '''

    if domain_name == "kitchen":
        color = f"""
        Additional constraints:
        1. In the kitchen domain, each PDDL object appears in the simulator as follows:
        - celery – green long rectangular block
        - radish – blue long rectangular block
        - bacon – magenta long rectangular block
        - egg – yellow long rectangular block
        - chicken – brown long rectangular block
        - apple – red long rectangular block
        - mysink – flat red rectangular prism on the right when viewed from the front
        - mystove – flat blue rectangular prism on the left when viewed from the front
        - mytable – the ground plane
        2. "cook" and "clean" are no-op actions with no effect: they cause no arm motion and do not change the scene configuration. They only change the symbolic PDDL predicates.
        3. Cooking a food on the stove right after cleaning on the sink is preferred.
        """
        system_prompt += "\n" + color
    return system_prompt


def vlm_backtrack_prompt(domain_name, problem, domain):

    system_prompt_gpt4o = f'''
        You are a replanning assistant for task-and-motion planning.
        Your job is to choose one visited leaf node in the hybrid state graph to backtrack to when planning to the current node has failed.

        Inputs provided:
        1. Simulator-Rendered Images for Current Node
        - Four views: front, top, left, and right.
        - These depict the execution of the edge’s action and its result.
        - Legend: Yellow (large frame): Initial end-effector pose before executing the action, Magenta (large frame): Desired pre-goal pose for the action, Blue (large frame): Desired pose after the action
        2. Constraint Violation Feedback: Describes why planning from the current node failed. 
    
        The initial state and goal are in the problem PDDL: {problem}
        Semantics of each predicate and action are in the domain PDDL: {domain}

        For your reasoning:
        1. Identify the failure from the feedback and images. Given the feedback and the images, reason which of the constraints below are violated and which node to backtrack to solve it.
        - Motion Planning Constraints: The robot failed to find a goal pose (grasp/place) without collision or a feasible motion path to it.
        - Kinematic Constraints: Extreme joint contortions or singularities appear.
        - Collision Constraints: Robot collides with unexpected objects during motion (expected collisions, such as between gripper and held object, are allowed).  
        - Grasp Constraints: (1) The object is not properly attached to the gripper, or (2) the object falls out of the gripper when tested in simulation.
        - Placement Constraints: The placed object moves significantly when the physics simulator is stepped after gripper release.  
        2. Evaluate alternate strategies:
        - If repeated attempts in the same region are likely to fail (e.g., cluttered workspace), choose a backtrack point that enables clearing or repositioning other objects first.
        - If the cause is limited space or blocked access, choose a node where alternative placement or stacking could have been chosen.
        3. Find the earliest corrective point: Among visited nodes, locate the earliest node whose effects, if replanned differently, could resolve the identified cause.
        - Please do not select the current node.

        Output Format
        1. First line: The exact name of the visited node you choose to backtrack to (must exist in the json). Do not include anything else on this line.
        2. Second line onward: A reasoning for your choice. 
        
        Additional constraints
        1. If information is chunked, combine across chunks before deciding.
        '''
    if domain_name == "kitchen":
        color = f"""2. In the kitchen domain, each PDDL object appears in the simulator as follows:
        - celery – green long rectangular block
        - radish – blue long rectangular block
        - bacon – magenta long rectangular block
        - egg – yellow long rectangular block
        - chicken – brown long rectangular block
        - apple – red long rectangular block
        - mysink – flat red rectangular prism on the right when viewed from the front
        - mystove – flat blue rectangular prism on the left when viewed from the front
        - mytable – the ground plane
        3. "cook" and "clean" are no-op actions with no effect: they cause no arm motion and do not change the scene configuration. They only change the symbolic PDDL predicates.
        """
        system_prompt_gpt4o += "\n" + color
    return system_prompt_gpt4o


def vlm_validator(domain_name, expected_predicates):
    domain_predicates = ""

    if domain_name == "tool_use":
        domain_predicates = """on(object1, object2): This predicate holds true when object1 is directly on top of object2, meaning they are in contact and object1 is above object2.
        holding(object): This predicate holds true when the robot's gripper is holding the specified object.
        arm-empty(): This predicate holds true when the robot's gripper is empty.
        on-table(object): This predicate holds true when the specified object is directly on top of the table/floor. Even if there are other objects stacked on top of it, on-table(object) can still be true as long as the bottom object is in contact with the table/floor.
        clear(object): This predicate holds true when there is no object on top of the specified object.

        Consistency rules:
        - holding(x) and arm-empty() cannot both be true.
        - If holding(x) is true, then on-table(x) must be false.
        - If on-table(x) is true, then holding(x) must be false.
        - If holding(x) is true, then on(x, y) must be false for any y, and on(y, x) must be false for any y.
        - If on(a, b) is true, then clear(b) must be false.
        - If clear(b) is true, then on(a, b) must be false for all a.
        """
        
    system_prompt = f'''
    Your goal is to answer questions related to object relationships in the given image(s) from the cameras of a Franka robot.
    An image from the front camera and an image from the left camera and an image from the right camera are provided for each question.

    We will use the following predicate-style descriptions to ask questions for the {domain_name} domain:
    {domain_predicates}

    You should respond 'Yes' or 'No'.

    Here are VLM predicates we have, note that they are defined over typed variables.
    Example: <predicate-name> (<obj1-variable> ...)

    Examples (separated by line or newline character):
    Do these predicates hold in the following images?
    1. on(red_block green_block)
    2. holding(blue_block)
    3. clear(red_block)
    4. on-table(green_block)

    Answer with explanation and Yes/No for each question. 
    Keep each explanation and answer in a single line, with no empty lines between responses:
    1. on(red_block green_block): The red block is stacked on the green block. [Yes]
    2. holding(blue_block): The blue block appears to be placed directly on the table, not being held by the robot. [No]
    3. clear(red_block): There is no object on top of the red block. [Yes]
    4. on-table(green_block): The green block is on the table. [Yes]


    Actual questions (separated by line or newline character):
    Do these predicates hold in the following images?
    {expected_predicates}

    Answer with explanation and Yes/No for each question. 
    Keep each explanation and answer in a single line, with no empty lines between responses:
    '''
    return system_prompt


def vlm_predicate_proposer(domain_name, initial_predicates, failed_predicates, objects):
    if domain_name == "tool_use":
        domain_predicates = """on(object1, object2): This predicate holds true when object1 is directly on top of object2, meaning they are in contact and object1 is above object2.
        holding(object): This predicate holds true when the robot's gripper is holding the specified object.
        arm-empty(): This predicate holds true when the robot's gripper is empty.
        on-table(object): This predicate holds true when the specified object is directly on top of the table/floor. Even if there are other objects stacked on top of it, on-table(object) can still be true as long as the bottom object is in contact with the table/floor.
        clear(object): This predicate holds true when there is no object on top of the specified object.

        Consistency rules:
        - holding(x) and arm-empty() cannot both be true.
        - If holding(x) is true, then on-table(x) must be false.
        - If on-table(x) is true, then holding(x) must be false.
        - If holding(x) is true, then on(x, y) must be false for any y, and on(y, x) must be false for any y.
        - If on(a, b) is true, then clear(b) must be false.
        - If clear(b) is true, then on(a, b) must be false for all a.
        """

    system_prompt = f'''
    You are given multi-view robot images (front, left, right) and:
    1. Initial Predicates with VLM evaluation results ([Yes] or [No])
    2. Failed Predicates - predicates that VLM marked as [No] and need refinement
    3. Objects - All objects in the scene

    Your task:
    Infer the final, complete, and logically consistent set of grounded predicates that are TRUE in the current scene.

    Current objects in the problem:
    {objects}

    Predicate definitions for the {domain_name} domain:
    {domain_predicates}

    Important instructions:
    - Treat initial predicates marked [Yes] as strong priors. Preserve them unless the images strongly suggest they are incorrect.
    - Identify the objects mentioned in Failed Predicates as primary target objects.
    - Infer predicates that are clearly true and visually supported.
    - You may include scene-level predicates such as arm-empty() if they are directly relevant and visually supported.
    - The final output must include:
    (a) initial [Yes] predicates that still hold,
    (b) corrected predicates replacing failed ones,
    (c) additional predicates that are clearly true and necessary to describe the scene.
    - Do not include predicates that are uncertain, redundant, or logically contradictory.

    Reasoning procedure:
    - Step 1: Identify the primary target objects from Failed Predicates.
    - Step 2: Read the initial predicate results as context.
    - Step 3: Re-check initial [Yes] predicates and keep only those still supported by the images.
    - Step 4: For each failed predicate, determine why it is false: wrong relation, wrong object pairing, wrong support/contact interpretation, wrong grasp state, etc.
    - Step 5: Infer corrected predicates for the target objects.
    - Step 6: Add any other predicates that are clearly true and necessary for a complete scene description.
    - Step 7: Check global consistency and remove contradictions.
    - Step 8: Output the final complete set of TRUE predicates in the current scene.

    Here are VLM predicates we have, note that they are defined over typed variables.
    Example: <predicate-name> (<obj1-variable> ...)

    Examples (separated by line or newline character):
    Initial Predicates and VLM evaluation results:
    1. on(red_block, green_block): The red block is stacked on the green block. [Yes]
    2. clear(red_block): There is no object on top of the red block. [Yes]
    3. holding(blue_block): The blue block appears to be placed directly on the table. [No]
    
    Failed Predicates:
    ['holding(blue_block)']

    Final TRUE predicates:
    1. on(red_block, green_block): red block is directly on top of the green block
    2. clear(red_block): no object on top of red block
    3. on-table(green_block): green block is supported by table
    4. on-table(blue_block): blue block is supported by table
    5. clear(blue_block): no object on top of blue block
    6. arm-empty(): gripper is not holding any object

    Actual input (separated by line or newline character):
    Initial Predicates and VLM evaluation results:
    {initial_predicates}

    Failed Predicates:
    {failed_predicates}

    Answer format:
    1. predicate(arguments): short explanation
    2. predicate(arguments): short explanation
    ...

    Based on the images, initial predicates, and failed predicates, return the final consistent set of all true predicates in the current scene.
    Keep each explanation and answer in a single line, with no empty lines between responses:
    '''
    return system_prompt