import numpy as np
class PCA:
    def __init__(self, n_component):
        self.components=None
        self.n_component=n_component
        self.mean=None
        self.eigval=None
        self.eigvec=None
    
    def fit(self, X):
        #calculate mean of X
        self.mean = np.mean(X,axis=0)

        #Centering of mean
        X = X - self.mean
        X=X/np.std(X,axis=0) #optional Step
        #calculate Covariance matrix
        cov= np.cov(X.T) # cov=np.matmul(X.T,X), similar
        
        #as in the documentation, the observations are in the columns of the observation

        #calculate eigenvalue eigenVector
        eigval, eigvec = np.linalg.eig(cov)

        #sort eigenvectors in decending order
        eigvec=eigvec.T # returned eigen vectors are column vectors
        indx= np.argsort(eigval)[::-1]
        self.eigval=eigval[indx]
        self.eigvec=eigvec[indx]
        

        #Store the first n eigenvectors to the self.component
        self.components=self.eigvec[0:self.n_component]

    def transform(self,X):
        X = X - self.mean
        return np.dot(X, self.components.T) #here we need the vector as columns vectors

    def variation(self,X):
        x,y=X.shape
        var=[]
        for i in range(self.n_component):
            #print(self.eigval[i])
            comp=self.eigval[i]/(x-1)
            var.append(comp)
        var=np.array(var)
        #print(var)
        var_sum=var.sum()
        #print(var_sum)
        var=var/var_sum
        #print(var)
        for j in range(self.n_component):
            print("PC"+str(j+1)," explains {} percent total variance".format(var[j]))
        print()
        print("Total variation explained by {} PC is {}".format(self.n_component,var.sum()))
        
