After cloning repo, to run app run these commands:
docker build -t test_ml_project .
docker run --rm -it test_ml_project

Run with container's shell
docker run --rm -it test_ml_project /bin/bash

Bind mount. To prevent having to rebuild image after every edit.
docker run --rm -it -v $(pwd):/app test_ml_project /bin/bash