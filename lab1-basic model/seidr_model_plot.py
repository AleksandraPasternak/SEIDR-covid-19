"""
based on https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
"""

import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class SEIDR:

    def __init__(self, incubation_period, infectious_period, N0, alpha, REP_0, avg_life_expectancy, E0, I0, D0, R0,
                 days):
        self.epsilon = 1. / incubation_period
        self.gamma = 1. / infectious_period
        self.N0 = N0
        # covid deaths rate
        self.alpha = alpha
        self.beta = (self.gamma + alpha) * REP_0
        self.u = 1. / avg_life_expectancy
        self.L = self.u * N0
        self.E0, self.I0, self.D0, self.R0 = E0, I0, D0, R0
        self.S0 = N0 - (E0 + I0 + R0 + D0)

        self.t = np.linspace(0, days, days * 100)  # dt = 0.01

    def derivatives_SEIDR(self, y, t):
        S, E, I, D, R = y
        dSdt = self.L - self.u * S - self.beta * S * (I / self.N0)
        dEdt = self.beta * S * (I / self.N0) - (self.u + self.epsilon) * E
        dIdt = self.epsilon * E - (self.gamma + self.u + self.alpha) * I
        dDdt = self.alpha * I
        dRdt = self.gamma * I - self.u * R
        return dSdt, dEdt, dIdt, dDdt, dRdt

    def calculate_model(self, incubation_period=None, infectious_period=None, E0=None, REP_0=None):
        if incubation_period is not None:
            self.epsilon = 1. / incubation_period
        if infectious_period is not None:
            self.gamma = 1. / infectious_period
        if E0 is not None:
            self.E0 = E0
        if REP_0 is not None:
            self.beta = (self.gamma + self.alpha) * REP_0

        # Initial conditions vector
        y0 = self.S0, self.E0, self.I0, self.D0, self.R0
        # Integrate the SIR equations over the time grid, t.
        solution = odeint(self.derivatives_SEIDR, y0, self.t)
        S, E, I, D, R = solution.T
        return S, E, I, D, R


if __name__ == '__main__':
    Lombardia_case = SEIDR(3, 8, 10000000, 0.006, 5.72, 82.8, 20000, 1, 0, 0, 80)
    S, E, I, D, R = Lombardia_case.calculate_model()

    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    axcolor = 'lightgoldenrodyellow'
    ax.margins(x=0)

    axepsilon = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    incubation_slider = Slider(
        ax=axepsilon,
        label='incubation period',
        valmin=2,
        valmax=12,
        valinit=3,
    )

    # Make a horizontal slider to control the frequency.
    axgamma = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    infectious_slider = Slider(
        ax=axgamma,
        label='infectious period',
        valmin=4,
        valmax=25,
        valinit=8,
    )

    # Make a vertically oriented slider to control the amplitude
    axR0 = plt.axes([0.05, 0.25, 0.0225, 0.63], facecolor=axcolor)
    REP_0_slider = Slider(
        ax=axR0,
        label="Reproduction ratio",
        valmin=0,
        valmax=15,
        valinit=5.72,  # beta=0.75
        orientation="vertical"
    )

    axE0 = plt.axes([0.15, 0.25, 0.0225, 0.63], facecolor=axcolor)
    E0_slider = Slider(
        ax=axE0,
        label="E0",
        valmin=1000,
        valmax=1000000,
        valinit=20000,
        orientation="vertical"
    )

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.3, bottom=0.3)

    t = Lombardia_case.t
    S, E, I, D, R = Lombardia_case.calculate_model()
    [line_S] = ax.plot(t, S / 1000000, 'b', alpha=0.5, lw=2, label='Susceptible')
    [line_E] = ax.plot(t, E / 1000000, 'c', alpha=0.5, lw=2, label='Exposed')
    [line_I] = ax.plot(t, I / 1000000, 'r', alpha=0.5, lw=2, label='Infected')
    [line_D] = ax.plot(t, D / 1000000, 'm', alpha=0.5, lw=2, label='Dead')
    [line_R] = ax.plot(t, R / 1000000, 'g', alpha=0.5, lw=2, label='Recovered')
    # plt.title('Lombardia case SEIDR')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of individuals (M)')
    # ax.set_ylim(0, 1000)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)


    def update(val):
        S, E, I, D, R = Lombardia_case.calculate_model(incubation_slider.val, infectious_slider.val, REP_0_slider.val,
                                                    E0_slider.val)
        line_S.set_ydata(S / 1000000)
        line_E.set_ydata(E / 1000000)
        line_I.set_ydata(I / 1000000)
        line_D.set_ydata(D / 1000000)
        line_R.set_ydata(R / 1000000)
        fig.canvas.draw_idle()


    # register the update function with each slider
    incubation_slider.on_changed(update)
    infectious_slider.on_changed(update)
    REP_0_slider.on_changed(update)
    E0_slider.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        incubation_slider.reset()
        infectious_slider.reset()
        REP_0_slider.reset()
        E0_slider.reset()


    button.on_clicked(reset)

    plt.show()

