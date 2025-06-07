# Virtual Environment Setup

This document explains how to automatically activate the backend virtual environment when working on this project.

## Option 1: Manual Activation (Simple)

The simplest method is to manually source the activation script:

```bash
source ./activate_backend.sh
```

This will activate the backend virtual environment for your current terminal session.

## Option 2: Automatic Activation with direnv (Recommended)

For automatic activation when you navigate to this project directory, you can use direnv:

1. Install direnv:
   - macOS: `brew install direnv`
   - Ubuntu/Debian: `sudo apt-get install direnv`

2. Add the following to your `~/.zshrc` file:
   ```bash
   eval "$(direnv hook zsh)"
   ```

3. Restart your terminal or run `source ~/.zshrc`

4. Allow the direnv configuration:
   ```bash
   direnv allow .
   ```

Now, whenever you navigate to this project directory, the backend virtual environment will be automatically activated.

## Option 3: Add to your .zshrc (Alternative)

If you don't want to use direnv, you can add the following to your `~/.zshrc` file:

```bash
# Auto-activate backend venv for deepschina project
function cd() {
  builtin cd "$@"
  
  # Get the absolute path of the current directory
  local current_dir=$(pwd)
  
  # Path to the deepschina project
  local project_path="/Users/jun77/Documents/Dropbox/a_root/code/deepschina"
  
  # Check if we're in the project directory or any subdirectory
  if [[ "$current_dir" == "$project_path"* ]]; then
    # Check if we're not already in the right virtualenv
    if [[ "$VIRTUAL_ENV" != "$project_path/backend/venv" ]]; then
      # Deactivate any active virtualenv
      if [[ -n "$VIRTUAL_ENV" ]]; then
        deactivate
      fi
      
      # Activate the backend virtualenv
      source "$project_path/backend/venv/bin/activate"
      echo "Backend virtual environment activated."
    fi
  elif [[ -n "$VIRTUAL_ENV" && "$VIRTUAL_ENV" == "$project_path/backend/venv" ]]; then
    # If we've left the project directory and the project's virtualenv is active, deactivate it
    deactivate
    echo "Left project directory. Virtual environment deactivated."
  fi
}
```

This will automatically activate the backend virtual environment when you navigate to the project directory or any of its subdirectories, and deactivate it when you leave.

## Verifying Activation

To verify that the virtual environment is activated, you can check which Python interpreter is being used:

```bash
which python
```

It should show: `/Users/jun77/Documents/Dropbox/a_root/code/deepschina/backend/venv/bin/python`
