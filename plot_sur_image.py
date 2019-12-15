
import pickle
from plot_result import plot

position_record = pickle.load( open( "log_SHINE/position_record_1212shownbgen", "rb" ) )
nfolder = 'log_SHINE/'
nimg = 'testimg'
#qtree = pickle.load( open( "log_SHINE/tree_record_121201_nbgen_300_sizepop_250", "rb" ) )

plot(position_record,nfolder,nimg,False)