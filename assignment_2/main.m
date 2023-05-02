% Clear workspace
clear
close(findall(0,'Type','figure'));
clc;


hold on
a = -1;
b = 1;
n_grid = 1025;
x = linspace(-1,1,n_grid);
y = f(x);
showFFT(y);
plot(x,y);
n_samples = 33;
t_s = linspace(-1,1,n_samples);
T_s = t_s(2) - t_s(1);
F_s = (1/(t_s(2) - t_s(1)));
y_s = f(t_s);

%stem(t_s,y_s,'filled','color','blue')

%reconstruction
x_ap = zeros(1,n_grid);
for x_i = 1:n_grid
    %convolution with sinc function
    for n = 1:n_samples
        x_ap(x_i) = x_ap(x_i) + y_s(n)*sinc(F_s*(x(x_i) - t_s(n)));
    end
end
xa = y_s * sinc( F_s * (ones(n_samples,1)*x - t_s' * ones(1 , n_grid) ) );
error = (max(abs(xa - y)));
plot(x,x_ap);

Q1(4,pi);

function y = f(x)
  y = cos(2*x) + sin(4*x);
  %y = exp(-1000*abs(x));
end


% 
% gif_obj = Gif('mile.gif');
% f = {};
% f{end+1} = createWave(0,0.5);
% f{end+1} = createWave(3,0.5*complex(0,1));
% f{end+1} = createWave(0.5,0.5*complex(0,1));
% bb(f);
% hold on
% axis(bb(f))
% axis equal
% scene(f,gif_obj);

function out = createWave(f,a)
t = linspace(0,2*pi,51);
out = a*exp(complex(0,f*t));
end
function scene(f,gif_obj)
m = length(f);
n = length(f{1});
for j = 1:n
    c = [0,0];
    fp = c;
    h = [];
for i = 1:m
    f_i = f{i};
    p_i = [real(f_i(j)),imag(f_i(j))];
    h(end+1) = plot(c(1) + real(f_i), c(2) + imag(f_i),'color','red');
    h(end+1) = quiver(c(1),c(2),p_i(1),p_i(2),'color','black');
    c = c + p_i;
end
plot(c(1),c(2),'o','markersize',5,'markerfacecolor','blue');
%gif_obj.update();
pause(0.2)
delete(h);
end
%gif_obj.update();
%gif_obj.save();
end
function out = bb(f)
m = length(f);
p = f{1};
for i = 2:m
    p = p + f{i};
end
p_max = [max(real(p)),max(imag(p))];
p_min = [min(real(p)),min(imag(p))];
c = (p_max + p_min)*0.5;
margin = 1.3;
p_max = margin*(p_max - c) + c;
p_min = margin*(p_min - c) + c;
out = [p_min(1),p_max(1),p_min(2),p_max(2)];
end

function Q1(Fs,T)
% if nargin == 1
%     L = Fs;
% end
%f = 12;
Ts = T/(Fs);
t = (0:Fs-1)*Ts;
%t_true = linspace(0,t(end),1000);
%y = cos(2*pi*f*t);
%y = cos(2*t) + sin(4*t);
y = f(t);
hold on
stem(t,y)

%plot(t_true,sin(2*pi*f*t_true));
%exportgraphics(gca,strcat('sinal','.jpeg'),'Resolution',333);

showFFT(y,t)
end
function c = showFFT(y,t)
N = length(y);
if nargin == 1
    t = [0,1/N];
end
FS = 1/(t(2) - t(1));
c = fft(y)'/N;
c = flip(fftshift(c));
x = (-N/2:N/2)*FS*2*pi/(N);
c_mag = [0;abs(c)];
phase = [0;angle(c)];

figure
hold on
stem(x,c_mag,'color','black');
ylabel('Amplitude')
xlabel('freq (Hz)')
exportgraphics(gca,strcat('amplitude','.jpeg'),'Resolution',333);

figure
hold on
stem(x,phase.*(c_mag > 1e-6),'color','black');
ylabel('Phase (rad)')
xlabel('freq (Hz)')
exportgraphics(gca,strcat('phase','.jpeg'),'Resolution',333);
end

function res = sinc(x)
res = sin(pi * x) ./ (pi * x);
end