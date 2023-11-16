
#Indiquer au fichier de conf containerd d'utiliser les registry des hosts qui sont dans "/etc/containerd/certs.d"
echo -e '\n[plugins."io.containerd.grpc.v1.cri".registry]\n   config_path = "/etc/containerd/certs.d"' | tee -a /etc/containerd/config.toml

#Inclure le registry-mirror d'orange dans les hosts pour docker.io
mkdir -p /etc/containerd/certs.d/docker.io 
echo -e 'server = "https://registry-1.docker.io"  # default after trying hosts\nhost."https://dockerproxy.repos.tech.orange".capabilities = ["pull", "resolve"]' | tee /etc/containerd/certs.d/docker.io/hosts.toml
systemctl restart containerd


