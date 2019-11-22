set -e
set -x

# mypy nglpy
flake8
coverage run --source nglpy setup.py test
deploy/test.sh

# if [[ ("$TRAVIS_BRANCH" == "master" && "$TRAVIS_PULL_REQUEST" == "false") || ! -z "$TRAVIS_TAG" ]]; then
#   ./deploy/upload.sh
# fi



