# Refer to `test.py` 

Just a test script to check out bazel builds.
Steps:
1. create python script: `tensorflow/models/inception/inception/test.py`
2. Edit bazel BUILD file (`tensorflow/models/inception/inception/BUILD`):
    ```python
    py_binary(
        name = "test",
        srcs = [
            "test.py",
        ],
        deps = [
            ":flowers_data",
            ":inception_eval",
        ],
    )
    ```
3. Build: `$ bazel build inception/test`
4. Run: `$ bazel-bin/inception/test`
