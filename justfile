dev:
    just fmt
    just lint

fmt:
    python3 -m ruff format

lint:
    python3 -m ruff check

gen-requirements:
    #!/bin/bash -ex
    # FIXME: https://github.com/bndr/pipreqs/issues/374
    # pipreqs --mode compat --force .
    ./scripts/pipreqs.py | tee requirements.txt
