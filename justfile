dev:
    just fmt
    just lint

fmt:
    python3 -m ruff format **/*.py

lint:
    python3 -m ruff check **/*.py

gen-requirements:
    #!/bin/bash -ex
    # FIXME: https://github.com/bndr/pipreqs/issues/374
    # pipreqs --mode compat --force .
    ./scripts/pipreqs.py | tee requirements.txt
