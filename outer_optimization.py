import numpy as np
import sparse as sp

def project_to_int( M_new):
	# this only projects to integer matrix and returns the matrix
    D,V = M_new.shape
    M_new_round = np.zeros((D,V))
    for v in range(V):
        
        diff = 0
        vals_index = [[0.0,0] for d in range(D)]
        for d in range(D):
            diff += M_new[d][v] - int(M_new[d][v])
            vals_index[d][0] = M_new[d][v]
            vals_index[d][i] = d

        vals_index = sorted(vals_index, key=lambda a:a[0], reverse = False)
        
        count=0
        while(diff>0.0):
            ind = vals_index[count][1]
            M_new_round[ind][v] = np.ceil(M_new[ind][v])
            diff -= (M_new_round[ind][v] - int(M_new[ind][v]))
            count+=1

    return M_new_round 



def calcGradient(eta, capital_phi, phi_star):
    D,V,K = capital_phi.shape
    eta_sum = [0.0]*K
    
    for k in range(K):
        for v in range(V):
            eta_sum[k] += eta[k][v]

    M_grad = np.zeros((D,V))
    d_arr, v_arr, k_arr = capital_phi.coords
    capital_phi = capital_phi.todense()
    print('Gradient Calculation started, total Items = ' + str(len(d_arr)))
   
    
    for i in range(len(d_arr)):
        d,v,k = [d_arr[i], v_arr[i], k_arr[i]]
        phi_kv = eta[k][v]/eta_sum[k]                
        M_grad[d][v] += (phi_kv - phi_star[k][v]) * ((eta_sum[k] - eta[k][v])/eta_sum[k]**2) * capital_phi[d, v, k]
        #if(i%1000 == 0): print('Done ' + str(i))
        
    print('Gradient Calculated')

    return M_grad



def update(eta, capital_phi, phi_star, M_0, M):
    D,V,K = capital_phi.shape
    L = 600
    L_d = 10

    #projection onto the set M
    M_grad = calcGradient(eta, capital_phi, phi_star)
    
    norm_1 = np.linalg.norm(M_grad, 1)    
    learning_rate = 10000
    print('Norm of gradient: ' + str(norm_1)) 
    if abs(norm_1)> 0.0001:
        learning_rate = (L - np.linalg.norm(M - M_0, 1))/norm_1

    for d in range(D):
        norm_1 = np.linalg.norm(M_grad[d], 1)
        if abs(norm_1)< 0.0001: continue 
            
        temp = (L_d - np.linalg.norm(M[d] - M_0[d], 1))/norm_1
        if temp < learning_rate:
            learning_rate = temp

    return M - (learning_rate*M_grad)

    
    
    


