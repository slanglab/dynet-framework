## Silver Parse Trees
These files were generated from the [BerkeleyParser](https://github.com/slavpetrov/berkeleyparser) directly from the Mikolov language modeling data using the command:
    
    cat (train/test/valid) | java -jar BerkeleyParser-1.7.jar -gr eng_sm6.gr -binarize > (train/test/valid).parse
    
Note that the language modeling data does not retain many original features of the text, so the performance is questionable.
I looked at a few of longer sentences and found the trees to be acceptable as a weak signal of syntactic structure, which may not be suitable for all purposes.
