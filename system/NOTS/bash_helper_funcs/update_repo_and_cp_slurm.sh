#!/bin/bash

# Default values
REPO_DIR="/projects/my13/kai/meta-pers-gest/fl-gestures"
SOURCE_FILEPATH="/projects/my13/kai/meta-pers-gest/fl-gestures/April_25/NOTS/first_run/maml_debug_run1.slurm"
DEST_FILEPATH="/scratch/my13/kai/runs/maml_debug_run1.slurm"  #Assuming this should be the filepath and not just the dir?
BRANCH="main"
RUN_SLURM="false"  # Default: don't run SLURM

# Parse named arguments
for arg in "$@"; do
  case $arg in
    --repo=*)
      REPO_DIR="${arg#*=}"
      ;;
    --source=*)
      SOURCE_FILEPATH="${arg#*=}"
      ;;
    --dest=*)
      DEST_FILEPATH="${arg#*=}"
      ;;
    --slurm=*)
      SLURM_FILE="${arg#*=}"
      ;;
    --branch=*)
      BRANCH="${arg#*=}"
      ;;
    --run-slurm=true)
      RUN_SLURM="true"
      ;;
    --run-slurm=false)
      RUN_SLURM="false"
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

echo "Repo directory: $REPO_DIR"
echo "Source file: $SOURCE_FILEPATH"
echo "Destination directory: $DEST_FILEPATH"
echo "Git branch: $BRANCH"
echo "Run SLURM: $RUN_SLURM"
[ -n "$SLURM_FILE" ] && echo "📝 SLURM file: $SLURM_FILE"

# Step 1: Navigate to repo and pull latest changes
cd "$REPO_DIR" || { echo "❌ Repo directory not found!"; exit 1; }
echo "📥 Pulling latest changes from origin/$BRANCH..."
git pull origin "$BRANCH"

# Step 2: Copy the file
echo "📤 Copying $SOURCE_FILEPATH to $DEST_FILEPATH..."
cp "$SOURCE_FILEPATH" "$DEST_FILEPATH" || { echo "❌ File copy failed!"; exit 1; }

# Step 3: Conditionally run SLURM job
if [ "$RUN_SLURM" = "true" ]; then
  echo "📡 Submitting SLURM job: $DEST_FILEPATH"
  sbatch "$DEST_FILEPATH"
else
  echo "✅ File copied. SLURM job not submitted."
fi
