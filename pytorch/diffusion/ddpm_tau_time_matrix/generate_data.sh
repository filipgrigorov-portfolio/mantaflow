N=$1
echo "Will generate $N simulation sequences"
for it in `seq 1 $N`; do
    echo "Generating sim $it"
    ../../../build/manta manta_generate_sim_fluid_data.py
done