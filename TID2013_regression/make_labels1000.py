import numpy as np
import multiprocessing 
from mpmath import mp
import pickle
from contextlib import closing

data = []

means = []
stds = []

filenames = []
with open('../datasets/TID2013/mos_with_names.txt', 'r') as f:
    for line in f:
        mos, filename = line.strip().split(" ")
        mos = float(mos)
        means.append(mos)
        filenames.append(filename)
        data.append([filename, mos]) #mean scores
        
i = 0
with open('../datasets/TID2013/mos_std.txt', 'r') as f:
    for line in f:
        mos_std = float(line.strip())
        mos_std_multiplied = mos_std*data[i][1]
        stds.append(max(mos_std_multiplied,0.1))
        data[i].append(mos_std) #standatd deviations
        data[i].append(mos_std_multiplied) #standatd deviations
        
        i += 1

hists = []

#mp.dps = 100

scores = np.array(range(100,1100))/100.



n_processes = 30
chunk_size = len(means)//n_processes


def worker(chunk_id):
    
    hists = []
    
    for idx in range(chunk_size*chunk_id, chunk_size*(chunk_id+1)):
        
        m = means[idx]
        
        m= m+1# shift scale to 1-10
        
        std = stds[idx] # I will think of these std values as relative

        pdf = lambda x: mp.npdf(x, m, std)
        
        hist = []
        for hist_bin in range(100,1100):
            hist.append(mp.quad(pdf, [hist_bin/100., (hist_bin+1)/100.]))
        
        
        # check difference (tails):
        hist_mean = mp.fsum([x*scores[i] for i, x in enumerate(hist) ])
        diff = m - hist_mean
        
        diff_divided = diff/(np.sum(np.array(range(100,1100))/100.))
        
        #diff_hist = [hist_mean_divided for idx, x in enumerate(hist)]
        
        # add remaining to hist
        hist = [x + diff_divided for x in hist]
        
        if idx%100==0:
            print('{} : {}/{}'.format(chunk_id,idx,chunk_size*(chunk_id+1)))
        
        # return to low precision
        hist = np.array([float(b) for b in hist], dtype=float)
        hists.append(hist)
    hists = np.vstack(hists)
    
    return hists



with closing(multiprocessing.Pool(processes=n_processes)) as pool:
        results = pool.map(worker, range(n_processes))


hists = np.vstack(results)
means = np.array(means)
stds = np.array(stds)
    

with open('tid2013_1000.pickle', 'wb') as f:
    pickle.dump({'hists':hists, 'means':means, 'stds':stds, 'filenames':filenames }, f)




np.set_printoptions(precision=3, suppress=True)
for i in range(5):
    print("mean: {}, std:{}, mean from hist:{} ".format(means[i]+1, stds[i],  hists[i].dot(scores.T)))
    
    #print("mean: {}, std:{}, mean from hist::{}, histogram:{}, ".format(means[i], stds[i],  hists[i].dot(scores.T), hists[i]))
        
   

'''
# generate gaussian histogram for each mean and std

plt.subplot(1, 2, 1)
out1 = plt.hist(means, 50, density=0, facecolor='green', alpha=0.5, range=(0,9))
plt.xticks(np.arange(0, 9, 1))

plt.xlabel('Scores')
plt.ylabel('No of images')
plt.title('Precise histogram of mean scores')
plt.grid(True)




# precise histogram of standard deviations

    

plt.subplot(1, 2, 2)
plt.hist(stds, 50, density=0, facecolor='green', alpha=0.5, range=(0,2.5))
plt.xticks(np.arange(0, 2.5, 0.5))

plt.xlabel('Scores')
plt.ylabel('No of images')
plt.title('Precise histogram of std devs')
plt.grid(True)

plt.show()
'''

