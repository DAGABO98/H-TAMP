find results/team_comp/logs -type f -name "*.err" -size +0c -print0 |
  while IFS= read -r -d '' f; do
    sig="$(grep -av '^[[:space:]]*$' "$f" | tail -n 1 | sed -E 's/[[:space:]]+/ /g')"
    printf '%s\t%s\n' "$sig" "$f"
  done |
  awk -F'\t' '
    {
      count[$1]++
      files[$1] = files[$1] "\n  " $2
    }
    END {
      for (s in count) {
        # print sortable header line: count<TAB>signature
        print count[s] "\t" s
      }
    }
  ' |
  sort -nr |
  while IFS=$'\t' read -r c s; do
    printf '%s\n' "============================================================"
    printf '%s\n' "$c file(s) with error signature:"
    printf '%s\n' "$s"
    printf '%s\n' "Files:"
    # re-compute the file list for this signature (second pass)
    find results/team_comp/logs -type f -name "*.err" -size +0c -print0 |
      while IFS= read -r -d '' f; do
        sig="$(grep -av '^[[:space:]]*$' "$f" | tail -n 1 | sed -E 's/[[:space:]]+/ /g')"
        [ "$sig" = "$s" ] && printf '  %s\n' "$f"
      done
    printf '\n'
  done
