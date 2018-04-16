import numpy as np
import sparse as sp

def project_to_int( M_new):
    '''
    This is only when we finish our iterations and have to project the final real
    matrix to an integer matrix
    '''

	# this only projects to integer matrix and returns the matrix
    D,V = M_new.shape
    M_new_round = np.int32(np.zeros((D,V)))
    for v in range(V):
        
        diff = 0
        vals_index = [[0.0,0] for d in range(D)]
        for d in range(D):
            diff += M_new[d][v] - int(M_new[d][v])
            vals_index[d][0] = M_new[d][v]
            vals_index[d][1] = d

        vals_index = sorted(vals_index, key=lambda a:a[0] - int(a[0]), reverse = True)
        
        count=0
        while(diff>0.0):
            ind = vals_index[count][1]
            M_new_round[ind][v] = np.ceil(M_new[ind][v])
            diff -= (M_new_round[ind][v] - int(M_new[ind][v]))
            count+=1

        while(count<D):
            ind = vals_index[count][1]
            M_new_round[ind][v] = int(M_new[ind][v])
            count +=1
            

    return M_new_round 



def calcGradient(eta, phi, fee, feestar, perm):
    D,V,K = phi.shape
    eta_sum = [0.0]*K
    eps = 0.005
    
    for k in range(K):
        for v in range(V):
            eta_sum[k] += eta[k][v]

    M_grad = np.zeros((D,V))
    d_arr, v_arr, k_arr = phi.coords
    phi = phi.todense()
    print('Gradient Calculation started, total Items = ' + str(len(d_arr)))

    
    for i in range(len(d_arr)):
        d,v,k1 = [d_arr[i], v_arr[i], k_arr[i]]
        k = k1
        
        '''
            the eps - l2 risk function
            this is actually no good. gives a zero gradient
        '''
        M_grad[d][v] += sgn(fee[k,v] - feestar[k1][v]) * max(0,(abs(fee[k,v] - feestar[k][v]) - eps))\
        *(1.0 - fee[k,v]) * phi[d, v, k]/eta_sum[k]

        '''
            the  simple l2 risk function
        '''

        #feestar has not been permuted
        #M_grad[d][v] += (fee[k,v] - feestar[k1][v])*(1.0 - fee[k,v]) * phi[d, v, k]


    print('Gradient Calculated')

    return M_grad

def sgn(a):
    if a>0.0:
        return 1
    else:
        return -1

def update(eta, phi, fee, feestar, M_0, M,it,momentum,perm = []):
    D,V,K = phi.shape
    if len(perm) == 0: perm = range(K)
    beta = 0.95
    L = 600
    L_d = 10
    flag = True
    #projection onto the set M
    M_grad = calcGradient(eta, phi, fee, feestar,perm)

    eps = 10**-15
    norm_1 = np.linalg.norm(M_grad, 1)
    print('Gradient: ' + str(norm_1)) 
    learning_rate = 1.0/eps
    
    if abs(norm_1)> eps:
        #to avoid division by zero
        learning_rate = (L - np.linalg.norm(M - M_0, 1))/norm_1

    '''
        Here i am calculating the learning rate 'lamda' such that the
        resultant 'M' lies in the constraint region that is: ||M_0 - M||_1< L and
        for all d: ||M_0[d] - M[d]||<L_d

        i take the minimum learning rate obtained from all these constraints
    '''
    for d in range(D):
        norm_1 = np.linalg.norm(M_grad[d], 1)
        if abs(norm_1)< eps:
            temp = 0.1
        #to avoid division by zero
        else:
        #temp learning_rate    
            temp = (L_d - np.linalg.norm(M[d] - M_0[d], 1))/norm_1

        M_grad[d] = M_grad[d]*min(temp*0.3, learning_rate*0.3)

##        for v in range(V):
##            if M_grad[d][v] < 0 :
##                M_grad[d][v] = M_grad[d][v]*min(temp*0.5, learning_rate*0.5)
##            else:
##                val = M_grad[d][v]*min(temp*0.2, learning_rate*0.2)
##                if M[d][v] - val < 0 :
##                    M_grad[d][v] = M[d][v]*0.5
##                else: M_grad[d][v] = val
            
##        if (temp) < learning_rate:
##            print("learning rate: %f , Norm: %f , doc No: %i , Norm of Ld - Norm: %f "%(abs(temp), norm_1, d, np.linalg.norm(M[d] - M_0[d],1)))
##            learning_rate = temp

    for d in range(D):
        norm_1 = np.linalg.norm(M[d] - M_0[d], 1)
        if norm_1 > L_d + eps:
            flag = False
            
    if (np.linalg.norm(M - M_0, 1)> L + eps):
        flag = False

    
    print('Assert all conditions satisified: ' + str(flag))
    print('learning rate into norm of gradient: ' + str(np.linalg.norm(M_grad, 1)))
    return (M - M_grad)*(M-M_grad>0)


##def update(eta, phi, fee, feestar, M_0, M,it,momentum,perm):
##    D,V,K = phi.shape
##    beta = 0.95
##    L = 600
##    L_d = 10
##
##    #projection onto the set M
##    M_grad = calcGradient(eta, phi, fee, feestar,perm)
##
##    eps = 10**-15
##    norm_1 = np.linalg.norm(M_grad, 1)
##    print('Gradient: ' + str(norm_1)) 
##    learning_rate = 1.0/eps
##    
##    if abs(norm_1)> eps:
##        #to avoid division by zero
##        learning_rate = 10 #(L - np.linalg.norm(M - M_0, 1))/norm_1
##
##    '''
##        Here i am calculating the learning rate 'lamda' such that the
##        resultant 'M' lies in the constraint region that is: ||M_0 - M||_1< L and
##        for all d: ||M_0[d] - M[d]||<L_d
##
##        i take the minimum learning rate obtained from all these constraints
##    '''
##    for d in range(D):
##        norm_1 = np.linalg.norm(M_grad[d], 1)
##        if abs(norm_1)< eps: continue
##        #to avoid division by zero
##
##        #temp learning_rate    
##        temp = (L_d - np.linalg.norm(M[d] - M_0[d], 1))/norm_1
##
##        
##        if (temp) < learning_rate:
##            print("learning rate: %f , Norm: %f , doc No: %i , Norm of Ld - Norm: %f "%(abs(temp), norm_1, d, np.linalg.norm(M[d] - M_0[d],1)))
##            learning_rate = temp
##
##    momentum = beta*momentum + (1-beta)*M_grad
##    M_grad = (M - ((50.0/(it))*momentum),momentum)
##    print('learning rate into norm of gradient: ' + str((50.0/(it))*np.linalg.norm(momentum, 1))) 
##    return (M_grad*(M_grad>0))

    
    
    



    
    
    


