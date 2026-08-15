import numpy as np
import matplotlib.pyplot as plt

def getCTF(x,w,t):
    Xw=np.zeros(len(w),dtype="complex")
    idx=0
    for wi in w:
        Xw[idx]=np.trapezoid(x*np.exp(-1*wi*t*1j),t)
        idx+=1
    return Xw


def MSE(x,y):
    x_mag=np.abs(x)
    y_mag=np.abs(y)

    x_phase=np.angle(x)
    y_phase=np.angle(y)

    return np.mean((x_mag-y_mag)**2), np.mean((x_phase-y_phase)**2)

#Time
T=8.0
N=40001
t=np.linspace(-T,T,N)
# Frequency axis (in Hz, since your signal has components at f = 4/(2*pi) and 6/(2*pi))
f = np.linspace(-2, 2, 2001)   # pick a range that covers your signal's frequencies
w = 2*np.pi*f                   # angular frequency ω = 2πf

x=0.5*np.cos(4*t)+0.5*np.sin(6*t)

#Derivatives
y1=-2*np.sin(4*t)+3*np.cos(6*t)
y2=-8*np.cos(4*t)-18*np.sin(6*t)
y3=32*np.sin(4*t)-108*np.cos(6*t)


Xw=getCTF(x,w,t)
Y1=getCTF(y1,w,t)
Y2=getCTF(y2,w,t)
Y3=getCTF(y3,w,t)

Y1ctf=(w)*Xw*1j
Y2ctf=w*w*Xw*1j*1j
Y3ctf=w*w*w*Xw*1j*1j*1j


mse_mag1,mse_phase1=MSE(Y1,Y1ctf)
print(f"For first derivative, mse in magnitude={mse_mag1}, and in phase={mse_phase1}")

mse_mag2,mse_phase2=MSE(Y2,Y2ctf)
print(f"For first derivative, mse in magnitude={mse_mag2}, and in phase={mse_phase2}")

mse_mag3,mse_phase3=MSE(Y3,Y3ctf)
print(f"For first derivative, mse in magnitude={mse_mag3}, and in phase={mse_phase3}")

fig, ax = plt.subplots()
ax.plot(f, np.abs(Y1), label="|Y1| direct")
ax.plot(f, np.abs(Y1ctf), '--', label="|Y1| theory")
ax.set_xlabel("f (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("First derivative")
ax.legend()


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(f, np.abs(Y2), label="|Y2| direct")
ax.plot(f, np.abs(Y2ctf), '--', label="|Y2| theory")
ax.set_xlabel("f (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title('Second derivative')
ax.legend()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(f, np.abs(Y3), label="|Y3| direct")
ax.plot(f, np.abs(Y3ctf), '--', label="|Y3| theory")
ax.set_xlabel("f (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title('Second derivative')
ax.legend()

plt.show()





