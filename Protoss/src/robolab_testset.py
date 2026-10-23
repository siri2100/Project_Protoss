"""Shared RoboLab testset for post-training and standalone evaluation."""

TEST_TASKS = (
    "AnimalsInBinTask",
    "AppleAndYogurtInBowlTask",
    "BagelsOnPlateTask",
    "BananaInBowlTask",
    "BananaOnPlateTask",
    "BananasInBinOneMoreTask",
    "BananasInBinThreeTotalTask",
    "BananasInCrateTask",
    "BananasOutOfBinTask",
    "BananaThenRubiksCubeTask",
)

# RoboLab runs num_envs episodes per sequential run: 10 tasks * 1 * 10 = 100.
TEST_NUM_ENVS = 1
TEST_NUM_RUNS = 10
