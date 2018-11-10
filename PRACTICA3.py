# -*- coding: utf-8 -*-
import matplotlib
#matplotlib.use('TkAgg')
from numpy import *
from scipy.sparse.linalg import cg,splu
from scipy.sparse import lil_matrix,identity
from matplotlib.pyplot import *
import time

def f0(x,y):
    #z=(5.0-4.0*(x**2+y**2))*exp(-(x**2+y**2))
    z=5.0*exp(-(x-0.5)**2-(y-0.5)**2)
    return z
def EJERCICIO1b(xi,xf,Nx,yi,yf,Ny,nu,u0,u1,u2,u3,fuente):
    tini=time.time()
    Nx=int(Nx)
    Ny=int(Ny)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dx2 = dx**2
    dy=(yf-yi)/float(Ny)
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    X,Y=meshgrid(x,y)
    Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),1)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),-1)
    My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)

    Mx[0,0]=0.0
    Mx[0,1]=0.0
    #Mx[Nx,Nx]=0.0
    Mx[Nx,Nx-1]=-2.0/dx2

    My[0,0]=0.0
    #My[Nx,Nx]=0.0

    for i in range(1,Ny):
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My


    A=Id+nu*A
    A=A.tocsc()

    b=zeros((Ny+1,Nx+1))
    b=fuente(X,Y)

    b[0,:]=u0*ones(Nx+1)

    b[Ny,:]=u2*ones(Nx+1)

    b[:,0]=u3*ones(Ny+1)

    b[1:Ny,Nx]+=2*nu/dx*u1*ones(Ny-1)

    b=b.reshape(N)

    LU=splu(A)
    usol=LU.solve(b)

    tfin=time.time()

    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20,cmap='jet')
    colorbar(cu)
    hold('on')
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fonsize=30)
    show()
    print "Tiempo de ejecucion:", tfin-tini

def EJERCICIO1b2(xi,xf,Nx,yi,yf,Ny,nu,u0,u1,u2,u3,fuente):
    tini=time.time()
    Nx=int(Nx)
    Ny=int(Ny)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My2=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    X,Y=meshgrid(x,y)
    Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),1)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),-1)
    My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)
    My2.setdiag(-2.0/(dy**2)*ones(Nx+1),0)

    Mx[0,0]=0.0
    Mx[0,1]=0.0
    Mx[Nx,Nx]=0.0
    Mx[Nx,Nx-1]=0.0

    #My2=2*copy(My)
    My[0,0]=0.0
    My[Nx,Nx]=0.0
    My2[0,0]=0.0
    My2[Nx,Nx]=0.0
    for i in range(1,Ny):
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My

    A[Ny*(Nx+1):(Ny+1)*(Nx+1),(Ny-1)*(Nx+1):(Ny)*(Nx+1)]=My2
    A[Ny*(Nx+1):(Ny+1)*(Nx+1),(Ny)*(Nx+1):(Ny+1)*(Nx+1)]=Mx

    A=Id+nu*A
    A=A.tocsc()

    b=zeros((Ny+1,Nx+1))
    b=fuente(X,Y)

    b[0,:]=u0*ones(Nx+1)

    b[Ny,:]+=2*nu/dy*u2*ones(Ny+1)


    b[:,0]=u3*ones(Ny+1)

    b[:,Nx]=u1*ones(Ny+1)

    b=b.reshape(N)

    LU=splu(A)
    usol=LU.solve(b)

    tfin=time.time()

    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20,cmap='jet')
    colorbar(cu)
    hold('on')
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fonsize=30)
    show()
    print "Tiempo de ejecucion:", tfin-tini

def EJERCICIO1(xi,xf,Nx,yi,yf,Ny,nu,u0,u1,u2,u3,fuente):
    tini=time.time()
    Nx=int(Nx)
    Ny=int(Ny)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dy=(yf-yi)/float(Ny)
    dx2 = dx**2
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    #Podriamos construir la identidad en vez de dimension
    #de dimension Nx+1 y trabajar por bloques
    Id=identity(N,dtype='float64',format='csc')
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    X,Y=meshgrid(x,y)
    Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),1)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),-1)
    My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)

    Mx[0,0]=0.0
    Mx[0,1]=0.0
    Mx[Nx,Nx]=0.0
    Mx[Nx,Nx-1]=0.0

    Mx[1,0]=0.0
    Mx[Nx-1,Nx]=0.0

    My[0,0]=0.0
    My[Nx,Nx]=0.0


    #Vamos a construir la matriz A por bloques
    #No voy a tener en cuenta ni la primera ni la ultima fila de bloques
    #Porque lo voy a dejar para despues y ver si tengo que cambiar algo o no
    for i in range(1,Ny):
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx #Este es el bloque diagonal
        if (i>1):
            A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My #Los bloques que estan a la izquierda de la diagonal
        if (i<Ny-1):
            A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My #Los bloques que estan a la derecha de la diagonal

    #Estos son los bloques de los bordes de la matriz A que son dependientes por la condiciones de contornos
    #A[(Nx+1):2*(Nx+1),0:(Nx+1)]=M
    #A[(Ny-1)*(Nx+1):Ny*(Nx+1),Ny*(Nx+1):(Ny+1)*(Nx+1)]=M

    A=Id+nu*A

    A=A.tocsc()

    b=zeros((Ny+1,Nx+1))
    b=fuente(X,Y)

    b[0,:]=u0*ones(Nx+1)
    b[1,1:Nx]+=nu*u0/(dy*dy)*ones(Nx-1)

    b[Ny,:]=u2*ones(Nx+1)
    b[(Ny-1),1:Nx]+=nu*u2/(dy*dy)*ones(Nx-1)

    b[:,0]=u3*ones(Ny+1)
    b[1:Ny,1]+=nu*u3/(dx*dx)*ones(Ny-1)

    b[:,Nx]=u1*ones(Ny+1)
    b[1:Ny,(Nx-1)]+=nu*u1/(dx*dx)*ones(Ny-1)
    #b[Ny,:]+=2*nu/dx*u1*ones(Nx-1)
    b=b.reshape(N)

    LU=splu(A)
    usol=LU.solve(b)

    tfin=time.time()

    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20,cmap='jet')
    colorbar(cu)
    hold('on')
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fonsize=30)
    show()
    print "Tiempo de ejecucion:", tfin-tini

def EJERCICIO2(xi,xf,Nx,yi,yf,Ny,T,Nt,nu,u0,u1,u2,u3,fuente):
    tini=time.time()
    Nx=int(Nx)
    Ny=int(Ny)
    Nt =int(Nt)
    T=float(T)
    xi=float(xi)
    xf=float(xf)
    yi=float(yi)
    yf=float(yf)
    dx=(xf-xi)/float(Nx)
    dx2 = dx**2
    dt = T/Nt
    dy=(yf-yi)/float(Ny)
    N=(Nx+1)*(Ny+1)
    A = lil_matrix((N,N), dtype='float64');
    Mx=lil_matrix((Nx+1,Nx+1),dtype='float64')
    My=lil_matrix((Nx+1,Nx+1),dtype='float64')
    Id=identity(N,dtype='float64',format='csc')
    #M =lil_matrix((Nx+1,Nx+1), dtype='float64')
    x=linspace(xi,xf,Nx+1)
    y=linspace(yi,yf,Ny+1)
    X,Y=meshgrid(x,y)
    Mx.setdiag(2.0*(1.0/(dx**2)+1.0/(dy**2))*ones(Nx+1),0)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),1)
    Mx.setdiag(-1.0/(dx**2)*ones(Nx+1),-1)
    My.setdiag(-1.0/(dy**2)*ones(Nx+1),0)

    Mx[0,0]=0.0
    Mx[0,1]=0.0
    #Mx[Nx,Nx]=0.0
    Mx[Nx,Nx-1]=-2.0/dx2

    My[0,0]=0.0
    #My[Nx,Nx]=0.0

    for i in range(1,Ny):
        A[i*(Nx+1):(i+1)*(Nx+1),i*(Nx+1):(i+1)*(Nx+1)]=Mx
        A[i*(Nx+1):(i+1)*(Nx+1),(i-1)*(Nx+1):i*(Nx+1)]=My
        A[i*(Nx+1):(i+1)*(Nx+1),(i+1)*(Nx+1):(i+2)*(Nx+1)]=My


    A=Id+nu*dt*A
    A=A.tocsc()

    ion()
    b=zeros((Ny+1,Nx+1))
    b=fuente(X,Y)

    b[0,:]=u0*ones(Nx+1)
    b[Ny,:]=u2*ones(Nx+1)
    b[:,0]=u3*ones(Ny+1)
    b[1:Ny,Nx]+=2*nu/dx*u1*ones(Ny-1)

    b=b.reshape(N)

    LU=splu(A)
    usol=LU.solve(b)
    usol2 = copy(usol)
    usol=0*usol.reshape((Ny+1,Nx+1))

    for i in range(Nt):
        b = dt*(fuente(X,Y)) + usol
        b[0,:]=dt*u0*ones(Nx+1)
        b[Ny,:]=dt*u2*ones(Nx+1)
        b[:,0]=dt*u3*ones(Ny+1)
        b[1:Ny,Nx]+=2*dt*nu/dx*u1*ones(Ny-1)
        b=b.reshape(N)
        usol=LU.solve(b)

        usol2 = copy(usol)
        usol=usol.reshape((Ny+1,Nx+1))
        clf()
        cu=contourf(X,Y,usol,20,cmap='jet')
        colorbar(cu)
        hold('on')
        cl=contour(X,Y,usol,20,colors='k')
        clabel(cl,inline=1,fonsize=30)
        pause(0.1)


    tfin=time.time()
    clf()
    usol=usol.reshape((Ny+1,Nx+1))
    cu=contourf(X,Y,usol,20,cmap='jet')
    colorbar(cu)
    hold('on')
    cl=contour(X,Y,usol,20,colors='k')
    clabel(cl,inline=1,fonsize=30)
    ioff()
    show()
    print "Tiempo de ejecucion:", tfin-tini

def EJERCICIO2Sofi(xi,xf,Nx,yi,yf,Ny,T,NT,nu,u0,u1,u2,u3,fuente):
    tini=time.time()
    ion()
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
    ioff()
    show()

#EJERCICIO1b2(0.0,1.0,100,0.0,1.0,100,2,0,0,0,0,f0)
EJERCICIO2(0.0, 1.0, 100, 0.0, 1.0, 100,2,20, 2.0, 0, 0, 0, 0, f0)
