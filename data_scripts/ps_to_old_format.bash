cat ../bfdb_with_s/indices.dat | while read i f; do
  echo $i _ $f
  ./ps_to_old_format.py ../bfdb_with_s/PS/$f.npz $i
done
