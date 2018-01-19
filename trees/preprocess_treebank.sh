PTB=penn_tb_3.0
OUTPUT_DIR=processed
SCRIPTS=treebank-scripts

export PERL5LIB=/home/jwei/ptb/treebank-scripts/
find $PTB -type d | grep '/MRG/WSJ/..' > sections

#apply transformations
while read fn; do
    basename=${fn##*/}
    perl $SCRIPTS/oneline -n $fn/*.MRG | \
        perl $SCRIPTS/fixsay | \
        perl $SCRIPTS/markargs | \
        perl $SCRIPTS/canonicalize | \
        perl $SCRIPTS/articulate | \
        perl $SCRIPTS/normcase -i | \
        perl $SCRIPTS/killnulls | \
        perl $SCRIPTS/binarize > $OUTPUT_DIR/$basename
done <sections

#extract trees
perl $SCRIPTS/selectsect 2 21 $OUTPUT_DIR/* | \
    perl $SCRIPTS/striplocations | perl $SCRIPTS/stripcomments > $OUTPUT_DIR/wsj_2-21.parse
perl $SCRIPTS/selectsect 23 23 $OUTPUT_DIR/* | \
    perl $SCRIPTS/striplocations | perl $SCRIPTS/stripcomments > $OUTPUT_DIR/wsj_23.parse
perl $SCRIPTS/selectsect 24 24 $OUTPUT_DIR/* | \
    perl $SCRIPTS/striplocations | perl $SCRIPTS/stripcomments > $OUTPUT_DIR/wsj_24.parse

#get text
cat $OUTPUT_DIR/wsj_2-21 | perl $SCRIPTS/fringe | perl $SCRIPTS/stripcomments > $OUTPUT_DIR/wsj_2-21.sent
cat $OUTPUT_DIR/wsj_23 | perl $SCRIPTS/fringe | perl $SCRIPTS/stripcomments > $OUTPUT_DIR/wsj_23.sent
cat $OUTPUT_DIR/wsj_24 | perl $SCRIPTS/fringe | perl $SCRIPTS/stripcomments > $OUTPUT_DIR/wsj_24.sent

#zip sentences and parses
