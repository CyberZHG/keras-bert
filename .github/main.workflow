workflow "Code Style" {
  on = "push"
  resolves = ["lint-action"]
}

action "lint-action" {
  uses = "CyberZHG/github-action-python-lint@master"
  args = "--max-line-length=120 keras_bert tests"
}
