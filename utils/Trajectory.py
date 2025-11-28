from sympy import solve, Symbol, poly
from math import sqrt


def min_parabolic_distance(a, b, c, x0, y0):
    '''
    Calcola la distanza tra un punto e una parabola
    Sia f(x)= ax^2+bx+c la parabola descritta
    Sia (x0, y0) un punto qualsiasi del piano
    Sia la distanza calcolata con la formula ((x-x0)^2 + (ax^2+bx+c-y0)^2)^0.5
    Argomenti:
        a:    coefficiente del termine di secondo grado
        b:    coefficiente del termine di primo grado
        c:    termine noto
        x0:   prima coordinata del punto su cui calcolare la distanza
        y0:   seconda coordinata del punto su cui calcolare la distanza
    Restituisce:
        La distanza minima in float del punto (x0, y0) dalla parabola y=ax^2+bx+c
    '''
    x = Symbol('x', real=True)  # variabile reale
    distance = poly((x - x0) ** 2 + (
                a * x ** 2 + b * x + c - y0) ** 2)  # formula della distanza (non tiene conto della radice quadrata)
    distance_derivate = poly(
        (x - x0) + (a * x ** 2 + b * x + c - y0) * (2 * a * x + b))  # derivata prima della distanza
    solutions = solve(distance_derivate, x)  # risultati della derivata prima uguale a zero

    # ricerca della distanza minima nei soli punti appartenenti ai numeri reali
    min_dist = float("inf")
    for point in solutions:
        if point.is_real:
            curr_dist = distance.subs({
                                          x: point}).evalf()  # sostituzione di un valore al posto della variabile e conversione in un numero float
            min_dist = curr_dist if curr_dist < min_dist else min_dist  # sostituzione minimo

    # per concordare con la formula la distanza deve tenere conto della radice quadrata
    return sqrt(min_dist)


def min_linear_distance(a, b, x0, y0):
    '''
    Calcola la distanza tra un punto e una retta
    Sia f(x)= ax+b la retta descritta
    Sia f(x) trasformata in ax-y+b=0
    Sia (x0, y0) un punto qualsiasi del piano
    Sia la distanza calcolata con la formula | ax0+by0+c | / sqrt( a^2+b^2 )
        b ha sempre valore -1
    Argomenti:
        a:    coefficiente del termine di primo grado
        b:    termine noto
        x0:   prima coordinata del punto su cui calcolare la distanza
        y0:   seconda coordinata del punto su cui calcolare la distanza
    Restituisce:
        La distanza minima in float del punto (x0, y0) dalla retta y=ax+b
    '''
    if a == 0:
        return abs(y0 - b)
    if a == 0 and b == 0:
        return abs(b)

    return abs(a * x0 - y0 + b) / a


def expected_distance(x0, y0, z0, init_velocity_x, init_velocity_y, init_velocity_z, init_x, init_y, init_z, g=9.81):
    '''
    Calcola la distanza attesa di un punto rispetto al un moto parabolico di un oggetto

    Sia (x0, y0, z0) un punto qualsiasi del piano
    Sia x(t)= init_velocity_x*t + init_x
    Sia y(t)= init_velocity_y*t + init_y
    Sia z(t)= -0.5g*t^2 + init_velocity_z*t + init_z

    Sia f(t)= [x(t), y(t), z(t)] l'insieme di formule del moto parabolico
    Siano calcolate le distanze prima rispetto al piano xy, xz; poi i valori sono combinati
    La distanza complessità è sqrt(xy_dist^2 + xz_dist^2)
    Argomenti:
        x0:               prima coordinata del punto su cui calcolare la distanza
        y0:               seconda coordinata del punto su cui calcolare la distanza
        z0:               terza coordinata del punto su cui calcolare la distanza
        init_velocity_x:  velocità iniziale rispetto all'asse x (dell'oggetto)
        init_velocity_y:  velocità iniziale rispetto all'asse y (dell'oggetto)
        init_velocity_z:  velocità iniziale rispetto all'asse z (dell'oggetto)
        init_x:           posizione iniziale rispetto all'asse x (dell'oggetto)
        init_y:           posizione iniziale rispetto all'asse y (dell'oggetto)
        init_z:           posizione iniziale rispetto all'asse z (dell'oggetto)
        g:                accelerazione gravitazionale
    Restituisce:
        La distanza minima in float del punto (x0, y0, z0) rispetto all'insieme di funzioni f(t)
    '''
    # caso init_velocity_x diverso da 0
    if (init_velocity_x != 0):
        # distanza nel piano xy
        # la traiettoria in questo piano è del tipo y= ax+b
        a = init_velocity_y / init_velocity_x
        b = init_y - a * init_x
        xy_dist = min_linear_distance(a, b, x0, y0)

        # distanza nel piano xz
        # la traiettoria in questo piano è del tipo y= ax^2+bx+c
        a = -g / (2 * init_velocity_x ** 2)
        b = init_velocity_z / init_velocity_x
        c = init_z - b * init_x
        xz_dist = min_parabolic_distance(a, b, c, x0, z0)

        result = sqrt(xy_dist ** 2 + xz_dist ** 2)

    # caso init_velocity_x uguale a 0
    else:
        # distanza rispetto all'asse x è costante
        x_dist = abs(x0 - init_x)

        # distanza nel piano yz
        # la traiettoria in questo piano è del tipo y= ax^2+bx+c
        a = -g / 2
        b = init_velocity_y
        c = init_z
        yz_dist = min_parabolic_distance(a, b, c, y0, z0)

        result = sqrt(x_dist ** 2 + yz_dist ** 2)

    return result


from math import sqrt

def expected_ball_drop_point(x0, y0, z0, vel_x, vel_y, vel_z, height=0.4, g=9.81, verbose=False):
    '''
    Calcola il punto atteso in cui la palla raggiungerà il tavolo
    Le formule utilizzate sono:
      x(t)= vel_x*t + x0
      y(t)= vel_y*t + y0
      z(t)= -1/2*g*t^2 + vel_z*t + z0
    Argomenti:
        x0:       posizione iniziale della palla rispetto all'asse x
        y0:       posizione iniziale della palla rispetto all'asse y
        z0:       posizione iniziale della palla rispetto all'asse z
        vel_x:    velocità iniziale della palla rispetto l'asse x
        vel_y:    velocità iniziale della palla rispetto l'asse y
        vel_z:    velocità iniziale della palla rispetto l'asse z
        heigth:   altezza per cui calcolare la posizione della pallina (rispetto all'asse z)
        g:        forza di gravità
        verbose:  se True mostra un messaggio su stdout del tempo utilizzato per il calcolo
    Restituisce:
        Una tripla (x,y,z) di float del punto di atterraggio atteso oppure None se all'altezza desiderata il punto non esiste
    '''
    delta= (vel_z)**2 - 4*(-0.5*g)*(z0-height)

    if( delta<0 ):
      return None

    time_add_positive= ( -(vel_z) + sqrt( delta ) ) / ( 2*(-0.5*g) )
    time_add_negative= ( -(vel_z) - sqrt( delta ) ) / ( 2*(-0.5*g) )

    time= max(time_add_positive, time_add_negative)
    if time < 0:
        return None
    if verbose:
      print(f"Time used: {time} sec")

    return ( float(vel_x*time+x0) , float(vel_y*time+y0) , float(height) )