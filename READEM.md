
mkdir -p ~/.ssh
chmod 700 ~/.ssh

curl -L https://github.com/xarnudvilas.keys >> ~/.ssh/authorized_keys

sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys