#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e

# This script substitutes environment variables in the /usr/share/nginx/html/index.html file.
# It's designed to run when the Nginx container starts.

HTML_FILE="/usr/share/nginx/html/index.html"

# Check if the target HTML file exists
if [ ! -f "$HTML_FILE" ]; then
    echo "ERROR: Target HTML file $HTML_FILE not found!"
    exit 1
fi

echo "Starting environment variable substitution in $HTML_FILE..."

# Get the VITE_API_BASE_URL from the container's environment variables.
# Provide a fallback default if it's not set, though it should be set in docker-compose.yml.
# The placeholder in index.html is ${VITE_API_BASE_URL}
TARGET_API_BASE_URL=${VITE_API_BASE_URL:-"http://localhost:8000/api_url_not_set_in_env"}

echo "  VITE_API_BASE_URL will be replaced with: ${TARGET_API_BASE_URL}"

# Substitute the placeholder in index.html.
# We use '#' as the sed delimiter to avoid issues if TARGET_API_BASE_URL contains '/'.
# The placeholder in index.html is literally "${VITE_API_BASE_URL}".
# We need to escape the '$' for sed when it's part of the pattern to match.
sed -i "s#\${VITE_API_BASE_URL}#${TARGET_API_BASE_URL}#g" $HTML_FILE

echo "Environment variable substitution complete."
echo "Nginx will now start serving the application."

# The original Nginx entrypoint script (or command) will execute next.
exit 0