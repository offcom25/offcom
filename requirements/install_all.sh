

set -ex

cd ./requirements



check_nvcc() {
    if command -v nvcc >/dev/null; then
        return 0
    else
        return 1
    fi
}

pip install pip --upgrade

find . -name '*.txt' | while read -r dep_file; do
    if [ -f "$dep_file" ]; then

        if [[ "$dep_file" == *"tensorrt"* ]] && ! check_nvcc; then
            echo "Skipping tensorrt"
            continue
        fi
        if [[ "$dep_file" == *"sys"* ]]; then

            pip install -r "$dep_file" --upgrade --pre
        else
            pip install -r "$dep_file"
        fi
    fi
done
