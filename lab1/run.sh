echo "basic cleanup.."
rm output
gcc vector_triad_arr.c -o vector_triad
echo "vector_triad_arr.c compiled.."
./vector_triad
echo "vector_triad calculations done and output file generated...."
echo "generating plot.."
python3 performance_plot.py
echo "cleanup.."
rm vector_triad
echo "cleanup done, now exiting"
