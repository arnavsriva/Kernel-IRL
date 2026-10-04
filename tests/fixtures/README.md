# Adversarial fixtures

Each fixture is a candidate kernel file with a header the tests read:

    # TASK: <task id>
    # EXPECT: correct | incorrect | hack_detected | static_reject | compile_error | timeout | crash
    # EXPECT_REASON: <taxonomy key>[|<alternative>...]   (optional)
    # SKIP_STATIC: true      (hostile only: proves runtime isolation holds if the static check is bypassed)
    # RUNTIME: false         (never execute; static verdict only)
    # NEEDS_BENCH: true      (the expected verdict only shows up in the benchmark stage)
    # NOTE: free text

Directories: good/ (must be correct with sane speedups), buggy/ (honest bugs -> incorrect),
hacks/ (reward hacks -> hack_detected or static_reject), hostile/ (sandbox must survive).
