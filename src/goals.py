"""
Global-goal (intention) texts per dataset.

Previously each bit_diffusion_* file had these dicts and picked one by
commenting lines in/out. Now the set is chosen from --dataset (or --goal_set).
"""

BREAKFAST_GOAL = {
    0: "the goal is to prepare scrambled eggs for breakfast",
    1: "the goal is to brew tea for breakfast",
    2: "the goal is to brew coffee for breakfast",
    3: "the goal is to pour milk for breakfast",
    4: "the goal is to prepare fried eggs for breakfast",
    5: "the goal is to prepare a bowl of cereal for breakfast",
    6: "the goal is to cook pancakes for breakfast",
    7: "the goal is to make a sandwich for breakfast",
    8: "the goal is to prepare a salad for breakfast",
    9: "the goal is to make juice for breakfast"
}

DARAI_GOAL = {
    0: "Using handheld smart devices",
    1: "Making a cup of instant coffee",
    2: "Making pancake",
    3: "Dining",
    4: "Cleaning dishes",
    5: "Making a cup of coffee in coffee maker",
    6: "Cleaning the kitchen",
}

SALADS_GOAL = {
    0: "Cut and mix ingredients",
    1: "Prepare dressing",
    2: "Serve salad",
    3: "Action end",
    4: "Action start",
}

EGTEA_GOAL = {
    0: "Prepare bacon and eggs",
    1: "Prepare a cheeseburger",
    2: "Prepare a continental breakfast",
    3: "Prepare a Greek salad",
    4: "Prepare a pasta salad",
    5: "Prepare a pizza",
    6: "Prepare a turkey sandwich",
}

# single-goal datasets: one text shared by the whole batch
UTKINECTS_GOAL = "the intention is to do the daily action"
NTURGBD_GOAL = "the intention is to do the daily action"


GOAL_SETS = {
    'darai': DARAI_GOAL,
    'breakfast': BREAKFAST_GOAL,
    '50salads': SALADS_GOAL,
    'egtea': EGTEA_GOAL,
    'utkinects': UTKINECTS_GOAL,
    'nturgbd': NTURGBD_GOAL,
}

# --dataset -> default goal set
DATASET_GOAL_SET = {
    'darai_l3': 'darai',
    'darai_l2': 'darai',
    'breakfast': 'breakfast',
    '50salads': '50salads',
    'utkinects': 'utkinects',
    'nturgbd': 'nturgbd',
}


def goal_texts(goal_set, goal_classes):
    """goal_classes: (B,) tensor of global-goal ids -> List[str] (or one str for single-goal sets)."""
    goals = GOAL_SETS[goal_set]
    if isinstance(goals, str):
        return goals
    return [goals[idx.item()] for idx in goal_classes]
