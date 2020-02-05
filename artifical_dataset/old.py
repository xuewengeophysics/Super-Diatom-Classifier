
# %% TEST 01 - SIMPLE KNN
# Constructing kd tree with known values
known = np.argwhere(global_patch_mask!=0)
kdt = KDTree(known, leaf_size=30, metric='euclidean')
# Finding neirest neighbors of unknownn values
unknown = np.argwhere(global_patch_mask==0)
nn_query = kdt.query(unknown, k=10, return_distance=True)
nn = nn_query[1]
nn_weights = nn_query[0]
# Filling

# %%
final_img = global_patch.copy()
showImg(final_img)
for i in range(len(unknown)):
    x_u, y_u = unknown[i][0], unknown[i][1] 
    avg_values = global_patch[known[nn[i]][:,0], known[nn[i]][:,1]]
    # final_img[x_u, y_u] = np.mean(global_patch[known[nn[i]][:,0], known[nn[i]][:,1]]).astype(np.uint8)
    final_img[x_u, y_u] = np.average(avg_values, weights=nn_weights[i]).astype(np.uint8)
showImg(final_img)
cv2.imwrite( "./yey.png", final_img)

# %% TEST 02 - COMPLEX KNN
final_img = global_patch.copy()
retval, labels = cv2.connectedComponents(global_patch_mask)
kdts = []
for i in range(1,retval):
    cc_indexes = np.argwhere(labels==i)
    tmp = KDTree(cc_indexes, leaf_size=30, metric='euclidean')
    kdts.append(tmp)