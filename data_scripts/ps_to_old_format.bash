cat bfdb/indices.dat | while read i f; do
  echo $i _ $f
  ./ps_to_old_format.py $f.npz $i
done
