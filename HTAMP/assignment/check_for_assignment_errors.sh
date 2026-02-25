#!/usr/bin/env bash
LOGS_ROOT="results/policies/logs"

# Iterate each immediate subdirectory of logs/
find "$LOGS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 |
while IFS= read -r -d '' dir; do
  # Skip if no non-empty .err files exist in this dir
  if ! find "$dir" -type f -name "*.err" -size +0c -print -quit | grep -q .; then
    continue
  fi

  printf '\n%s\n' "################################################################"
  printf '%s\n' "Folder: $dir"
  printf '%s\n' "################################################################"

  find "$dir" -type f -name "*.err" -size +0c -print0 |
    while IFS= read -r -d '' f; do
      sig="$(grep -av '^[[:space:]]*$' "$f" | tail -n 1 | sed -E 's/[[:space:]]+/ /g')"
      printf '%s\t%s\n' "$sig" "$f"
    done |
    awk -F'\t' '
      { count[$1]++ }
      END { for (s in count) print count[s] "\t" s }
    ' |
    sort -nr |
    while IFS=$'\t' read -r c s; do
      printf '%s\n' "============================================================"
      printf '%s\n' "$c file(s) with error signature:"
      printf '%s\n' "$s"
      printf '%s\n' "Files:"

      # second pass: list files for this signature within THIS dir
      find "$dir" -type f -name "*.err" -size +0c -print0 |
        while IFS= read -r -d '' f; do
          sig="$(grep -av '^[[:space:]]*$' "$f" | tail -n 1 | sed -E 's/[[:space:]]+/ /g')"
          [ "$sig" = "$s" ] && printf '  %s\n' "$f"
        done
      printf '\n'
    done
done