declare -A ips
for key in $(yq eval 'keys | .[]' ../../config/clus_params.yaml); do
    docker exec $key-control-plane cat /etc/kubernetes/admin.conf > ../../config/cls_contexts/$key-control-plane.yaml
    result1=$(docker exec $key-control-plane ip a | grep -A 2 'eth0@' | grep -oP 'inet \K[\d./]+')
    result2=$(docker exec $key-worker ip a | grep -A 2 'eth0@' | grep -oP 'inet \K[\d./]+')
    ips[$key]=$(printf '{"control-plane": "%s", "worker": "%s"}' "$result1" "$result2" | jq -c)
done

# Save the dictionary to a file in JSON format
printf '{' > ../../config/clus_ips.json
for key in "${!ips[@]}"; do
    printf '"%s": %s,' "$key" "${ips[$key]}" >> ../../config/clus_ips.json
done
sed -i '$ s/,$//' ../../config/clus_ips.json # remove trailing comma
printf '}' >> ../../config/clus_ips.json