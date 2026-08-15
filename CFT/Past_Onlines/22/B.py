import numpy as np
import matplotlib.pyplot as plt

class Signal_Generator:
    def getGaussian(self,a,t):
        return np.exp(-1*(a*(t*t))) 


def getCFT(x,w,t):
    Xw=np.zeros(len(w),dtype="complex")
    idx=0
    for wi in w:
        Xw[idx]=np.trapezoid(x*np.exp(-1*wi*t*1j),t)
        idx+=1
    return Xw


def MSE(x, y, f, t0):
    x_mag, y_mag = np.abs(x), np.abs(y)
    x_phase, y_phase = np.angle(x), np.angle(y)

    predicted_phase = x_phase - 2*np.pi*f*t0
    phase_error = np.angle(np.exp(1j*(y_phase - predicted_phase)))
    mse_mag = np.mean((x_mag - y_mag)**2)
    mse_phase = np.mean(phase_error**2)
    return mse_mag, mse_phase

t=np.linspace(-5,5,2001)
f = np.linspace(-10, 10, 1001)
w = 2*np.pi*f 


signal_Generator=Signal_Generator()

x=signal_Generator.getGaussian(1,t)
y=signal_Generator.getGaussian(1,t-1)


x_cft=getCFT(x,w,t)
y_cft=getCFT(y,w,t)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(f, np.abs(x_cft), label="X")
ax.plot(f, np.abs(y_cft), '--', label="Y")
ax.set_xlabel("f (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title('CFT of x and y')
ax.legend()


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(f, np.angle(x_cft), label="X")
ax.plot(f, np.angle(y_cft), '--', label="Y")
ax.set_xlabel("f (Hz)")
ax.set_ylabel("Phase")
ax.set_title('CFT of x and y')
ax.legend()



mse_mag,mse_phase=MSE(x_cft,y_cft,f,1)

print(f"{mse_mag} {mse_phase}")

plt.show()