import numpy as np
from scipy import interpolate, integrate, stats
import matplotlib.pyplot as plt

DURATION = 0.5 #[sec]
TIMESTEPS = 100

def ODE_system(t,state,a,b,C,alpha0,beta0,m1,m2):

    alpha, alpha_dot = state

    alpha0 = np.radians(alpha0)
    beta0 = np.radians(beta0)

    L1 = a/100
    L2 = b/100

    L1g = 0.5*L1
    L2g = 0.5*L2

    J1 = 1/12*m1*L1**2
    J2 = 1/12*m2*L2**2

    R = (np.cos(alpha))/(np.cos(beta0))

    P1 = L2g*np.sin(alpha0)*np.sin(beta0)-L1*(np.sin(alpha0)**2)-(L2g**2/L2)*R*(np.sin(beta0)**2)+2*(L1*L2g/L2)*R*np.sin(alpha0)*np.sin(beta0)-\
        (L1**2/L2)*R*(np.sin(alpha0)**2)
    P2 = L2g*np.cos(beta0)-L1*np.cos(alpha0)-(L2g**2/L2*np.cos(beta0))+2*(L1*L2g/L2)*np.cos(alpha0)-(L1**2/L2)*R*np.cos(alpha0)
    
    K1 = L2g*np.cos(beta0)*np.sin(alpha0)-L1*np.cos(alpha0)*np.sin(beta0)-(L2g**2/L2)*np.cos(alpha0)*np.sin(beta0)+\
        (L1*L2g/L2)*R*np.cos(alpha0)*np.sin(alpha0)+(L1*L2g/L2)*np.cos(alpha0)*np.sin(alpha0)-(L1**2/L2)*R*np.cos(alpha0)*np.sin(alpha0)
    K2 = L2g*np.sin(beta0)-L1*np.sin(alpha0)-(L2g**2/L2)*np.sin(beta0)+(L1*L2g/L2)*(np.sin(alpha0)+R*np.sin(beta0))-(L1**2/L2)*R*np.sin(alpha0)
    
    N1 = np.cos(alpha0)-(L2g/L2)*np.cos(alpha0)+(L1/L2)*R*np.cos(alpha0)
    N2 = ((np.sin(alpha0))**2)-(L2g/L2)*R*np.sin(alpha0)*np.sin(beta0)+(L1/L2)*R*(np.sin(alpha0))**2
    N3 = np.sin(alpha0)*np.cos(alpha0)-(L2g/L2)*R*np.cos(alpha0)*np.sin(beta0)+(L1/L2)*R*np.cos(alpha0)*np.sin(alpha0)
    N4 = (L2g/L2)*np.sin(alpha0)-np.sin(alpha0)-(L1/L2)*R*np.sin(alpha0)

    term1 = C+alpha_dot**2*(L1/L2)**2*R/(np.cos(beta0)**2)*((np.cos(alpha0)**2)*np.tan(beta0)-np.sin(beta0)*np.sin(alpha))*(J2-(P1+P2)*L2)-\
        m2*L1*alpha_dot**2*((L1/L2)*R)**2*(K1+K2*np.cos(alpha0))-m2*L1**2*alpha_dot**2*(N3-N4*np.cos(alpha0))
    
    term2 = m1*L1g**2+J1+J2*((L1/L2)*R)**2-m2*(L1**2/L2)*R*(P1+P2)+m2*L1**2*(N1*np.cos(alpha0)+N2)

    f = term1/term2

    alpha_ddot = f

    return [alpha_dot, alpha_ddot]

def linear_model(t,initial_data,angle):
    data_for_regr = initial_data-angle
    
    slope, intercept, r_value, pvalue, std_err = stats.linregress(t, data_for_regr)
    
    intercept = intercept+angle
    model_prediction = intercept+slope*t

    plt.scatter(t, initial_data, label='Experimental' ,s=7.5)
    plt.plot(t,model_prediction, label='Linear Regression', color='red')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    return [np.radians(slope), r_value, std_err]

def angles_calc(a,b,alpha0,beta0,alpha_d,t):
    alpha = np.zeros(TIMESTEPS)
    beta = np.zeros(TIMESTEPS)
    beta_d = np.zeros(TIMESTEPS)
    t_int = np.zeros(TIMESTEPS)
    beta_d_int = np.zeros(TIMESTEPS)

    a = a/100
    b = b/100

    alpha[0] = np.radians(alpha0)
    beta[0] = np.radians(beta0)
    
    beta_d[0] = -alpha_d*(a/b)*(np.cos(alpha[0]))/(np.cos(beta[0]))

    for k in range(1,TIMESTEPS):
        alpha[k] = alpha[0] + alpha_d*t[k]          #alpha_d is considered constant
    
        #Creations of arrays of values used to compute the integral of beta_dot with respect to time    

        beta_d_int[0] = beta_d[0]
        t_int[0] = t[0]
        for j in range(k-1):
            beta_d_int[j+1] = beta_d[j]
            t_int[j+1] = t[j]

        #Since we don't know the relationship or the analytical expression of beta and beta_dot 
        #we calculate beta at step k via numerical integration

        integral_res = integrate.cumtrapz(beta_d_int, t_int, initial=0)
        beta[k] = beta[0]+integral_res[k]

        beta_d[k] = -alpha_d*(a/b)*(np.cos(alpha[k]))/(np.cos(beta[k]))
    
    
    xi = np.degrees(beta-np.pi-alpha) #[°]
    beta = np.degrees(beta)
    return [beta, xi]
    

def main():

    #measurements are just for testing, need to be anatomically verified

    t = np.linspace(0,DURATION,TIMESTEPS)
    
    # From anthropometric tables we compute the lenghts and masses of the phalanxes (mean lenght of middle and distal phalanges of index finger)
    # From anthropometric tables we compute the depths (diameters if cylindrical) of the middle and distal phalanges of index finger
    # Human bone bulk density 1.8-2.1 g/cm^3 -> assuming bulk density is constant throughout the volume and the phalanges are cylindrical

    L1 = 3.42 #[cm]
    L2 = 1.95 #[cm]
    m1 = 1.95*np.pi*((1.89/2)**2)*L1
    m2 = 1.95*np.pi*((1.59/2)**2)*L2
    torque_motor = 4.905 #[Nm], torque applied by the motor
    angle1 = 45 #[°]
    angle2 = 315 #[°]
    angle3 = angle2-180-angle1 #[°]

    initial_cond = np.array([angle1, 0])
    t_span = (0,DURATION)
    solution = integrate.solve_ivp(ODE_system, t_span, initial_cond, t_eval=t, args=(L1,L2,torque_motor, angle1, angle2, m1, m2), method='RK45')

    alpha = solution.y[0]
    alpha_dot, r_value_alpha, alpha_err = linear_model(t,alpha,angle1)

    print(f"R^2 for ALPHA (CRANK ANGLE): {(r_value_alpha**2)*100}%") 
    print(f"With STANDARD ERROR (ALPHA): {round(alpha_err,5)}")
    print(f"According to the linear model, the angular velocity ALPHA DOT is: {round(alpha_dot,5)} deg/s")

    beta = angles_calc(L1,L2,angle1,angle2,alpha_dot,t)[0]     
    beta_dot, r_value_beta, beta_err = linear_model(t,beta,angle2)

    print(f"R^2 for BETA (CONNECTING ROD ANGLE): {(r_value_beta**2)*100}%")
    print(f"With STANDARD ERROR (BETA): {round(beta_err,5)}") 
    print(f"According to the linear model, the angular velocity BETA DOT is: {round(beta_dot,5)} deg/s")

    xi = angles_calc(L1,L2,angle1,angle2,alpha_dot,t)[1]
    xi_dot, r_value_xi, xi_err= linear_model(t,xi,angle3)

    print(f"R^2 for XI (ANGLE ON REVOLUTE JOINT BETWEEN CRANK AND CONNECTING ROD): {(r_value_xi**2)*100}%") 
    print(f"With STANDARD ERROR (XI): {round(xi_err,5)}")
    print(f"According to the linear model, the angular velocity XI DOT is: {round(xi_dot,5)} deg/s")

main()