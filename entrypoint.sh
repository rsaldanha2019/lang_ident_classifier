#!/bin/bash
set -e

USER_ID=${UID:-1000}
GROUP_ID=${GID:-1000}
USER_NAME=${USERNAME:-appuser}

# Try to get group name by GID
GROUP_NAME=$(getent group "$GROUP_ID" | cut -d: -f1)

# If no group with this GID exists, create one
if [ -z "$GROUP_NAME" ]; then
    groupadd -g "$GROUP_ID" "$USER_NAME"
    GROUP_NAME="$USER_NAME"
fi

# If user doesn't exist, create with matching UID/GID
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
    useradd -m -u "$USER_ID" -g "$GROUP_ID" -s /bin/bash "$USER_NAME"
fi

# Own the /app folder so volume mounts are writable
chown -R "$USER_ID:$GROUP_ID" /app

# Run command as the created user
exec gosu "$USER_NAME" "$@"
