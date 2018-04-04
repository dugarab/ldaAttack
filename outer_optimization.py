def project_to_int( M_new):
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
            diff -= M_new_round[ind][v] - int(M_new[ind][v])
            count+=1

    return M_new_round 



def calcGrandient(eta, delta, capital_phi, phi_star):
    D,V,K = phi.shape
    eta_sum = [0.0]*K
    
    for k in range(K):
        eta_sum[k] += sum(eta[k])

    M_grad = np.zeros(D,V)
    
    for d in range(D):
        for v in range(V):
            m_sum = 0.0
            for k in range(K):
                phi_kv = eta(k,v)/eat_sum(k)
                
                m_sum += (phi_kv - phi_star[k][v]) *  \
                         ((eta_sum(k) - eta[k][v])/eta_sum(k)**2) * \
                         capital_phi[d][v][k]
            M_grad[d][v] += m_sum

    return M_grad



def update(eta, delta, capital_phi, phi_star, M_0, M):
    
    L = 600
    L_d = 10

    #projection onto the set M
    M_grad = calcGradient(eta, delta, capital_phi, phi_star)

    learning_rate = (L - np.linalg.norm(M - M_0, 1))/np.linalg.norm(M_grad, 1)

    for d in range(D):
        temp = (L_d - np.linalg.norm(M[d] - M_0[d], 1))/np.linalg.norm(M_grad[d], 1)
        if temp < learning_rate:
            learning_rate = temp

    M -= learning_rate*M_grad

    return M
    
    
    


