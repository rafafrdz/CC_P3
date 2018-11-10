# -*- coding: utf-8 -*-
# Ejercicio 3 

# -*- coding: utf-8 -*-
# Ejercicio 2

# -*- coding: utf-8 -*-
import matplotlib
#matplotlib.use('TkAgg')
from numpy import *
from scipy.sparse.linalg import cg,splu
from scipy.sparse import lil_matrix,identity
from matplotlib.pyplot import *

import time



def f0(x,y,t):
    #z=(5.0-4.0*(x**2+y**2))*exp(-(x**2+y**2))
    z=5.0*exp(-(x-0.5)**2-(y-0.5)**2)
    return z

def ui(x,y):
    z=0*x
    return z

    
def eliptico_2d_dirichlet_no_simetrico(xi,xf,Nx,yi,yf,Ny,nu,u0,u1,u2,u3,fuente,T,NT):  
    tini=time.time()
    Nx=int(Nx)
    Ny=int(Ny)
    NT=int(NT)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    T=float(T)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny) 
    dt=T/float(NT)   
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    t=linspace(0.0,T,NT+1) 
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1) # Partición en X
    y=linspace(yi,yf,Ny+1) # Partición en Y
    X,Y=meshgrid(x,y) # Producto tensorial de X e Y (mallado o rejilla)(generea el conjunto de pares que me definen mi rejilla)
    Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),1)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),-1)
    My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)
       
    Mx[0,0]=0.0 #Imponiendo las condiciones de dirichlet para Mx y My (por nuestra numeración en las aristas verticales)
    Mx[0,1]=0.0
   # Mx[Nx,Nx]=0.0
    Mx[Nx,Nx-1]=-2.0/(dx**2)   
        
    My[0,0]=0.0
    #My[Nx,Nx]=0.0
    
    for i in range(1,Ny): # Tiene Ny+1 filas pero como la primera y la última se ve afectada por los problemas de contorno solo tomamos de 1 a Ny
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx   # El operador : coge donde empiezo y donde acabo(para recorer solo los índices necesarios)
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My 
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My
    
    
    A=Id+nu*dt*A
    A=A.tocsc()
    LU=splu(A)
    
    usol = 0*X
    
    for i in range(0,NT):
        b=usol + dt*fuente(X,Y,(i+1)*dt)
        b[0,:]=u0*ones(Nx+1) # Fila 0
        b[Ny,:]=u2*ones(Nx+1) # Fila Ny
        b[:,0]=u3*ones(Ny+1) # Columna 0
        b[1:Ny,Nx]=+ u1*ones(Ny-1)*2*dt*nu/dx # Columna Nx
        b=b.reshape(N) # Reordena la matriz (fila 0 y lo pone como vector luego la siguiente fila y pone debajo) para dormar mi rejilla
        usol=LU.solve(b)
        usol=usol.reshape((Ny+1,Nx+1))
        clf() # Borramos y pintamos lo siguiente
        cu=contourf(X,Y,usol,20,cmap='jet') # Curva nivel rellena
        colorbar(cu)
        hold('on')
        cl=contour(X,Y,usol,20,colors='k') # Separa las líneas de colores con una línea negra
        clabel(cl,inline=1,fonsize=30) # Lo dibuja
        title("tiempo:"+str((i+1)*dt)) # Para ver por qué tiempo voy
        show()
        pause(0.1) # Paramos un segundo para hacer la animación
   
   
     # Vuelvo para atrás y lo transformo en una rejilla
    
   
    