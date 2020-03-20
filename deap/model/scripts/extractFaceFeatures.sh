folder_counter=0
for folder in ../videos/*;
 do
   for file in $folder/*;
    folder_counter=$((folder_counter+1))
      if [[ "$counter" -gt 2 ]]; then
        echo "Counter: $counter times reached; Waiting loop!"
#        wait
#        counter=0
      fi
    do
      echo $folder
      #/usr/local/bin/FeatureExtraction -f $file &
      if [[ "$counter" -gt 2 ]]; then
        echo "Counter: $counter times reached; Waiting loop!"
#        wait
#        counter=0
      fi

    done;
  wait
done

