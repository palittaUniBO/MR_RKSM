% Example 6.2 in
%
% Minimal Residual Rational Krylov Subspace Method for Sequences of Shifted Linear Systems
% Hussam Al Daas and Davide Palitta 
% ArXiv: 2507.00267
%
% Please, acknowledge our work by citing our manuscript whenever you use the software 
% provided here in your research.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
% INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
% PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
% HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
% OTHER DEALINGS IN THE SOFTWARE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all;
addpath(genpath('../src'))


A=mmread('qc2534.mtx');
n1 = length(A);
k = 1;
% number of shifts
rng('default')
C = randn(n1,k);
C=C/norm(C);


% maximum number of iterations
iter = 150;

% convergence tolerance
tol = 1e-8;


disp('*** Shifts: Complex - no conjugate pair')

center = -0.8 - 0.07*1i;
vscale = 0.1;
radius = 0.2;

for n2=[256 512 1024]
    fprintf('************\n ell=%d \n************\n',n2)
    D = kron(ones(n2,1),eye(k));
    normb = sqrt(abs(diag(D*(C'*C)*D')));

    s = zeros(n2,1);
    for i = 1 : n2
        theta = 2.0*pi*(i)/n2;
        s(i) = center + radius* (cos(theta)+1i*vscale*sin(theta));
    end
    % the matrix with shifts
    B = diag(sparse(s));

    %% Rational KryloINF

    params.m=iter;
    params.tol=tol;
    params.shift_init=s(1);
    params.ch=0;

    tt=tic;
    [U,V,it,res,shifts,RES_tot] = rational_ksm_shifted_systems(A,s,C,params);
    time_RKSM=toc(tt);

    BB=kron(B,speye(k));
    Z=U*V;
    RES=A*Z+Z*BB-C*D';
    normr=zeros(n2,1);
    for t=1:n2*k
        normr(t)=norm(RES(:,t));
    end  

    fprintf("Rational Krylov\n max (column-wise) 2-norm of relative residuals: %e\n",max(normr./normb));
    fprintf("Iterations: %d, Solution rank: %d, Execution time: %e\n", it,size(U,2),time_RKSM);


    params.maxit=params.m;
    params.poles='ADM';
    params.real=0;
    tt=tic;
    [U,V,its,nrmrestotnew,RES2] = rk_adaptive_sylvester_LS(A, s, C, params);
    time_RKSM2=toc(tt);

    Z=U*V;
    RES=A*Z+Z*BB-C*D';
    normr=zeros(n2,1);
    for t=1:n2*k
        normr(t)=norm(RES(:,t));
    end  

    disp('********')
    fprintf("Rational Krylov (ADM poles) \n max (column-wise) 2-norm of relative residuals: %e\n",max(normr./normb));
    fprintf("Iterations: %d, Solution rank: %d, Execution time: %e\n", its,size(U,2),time_RKSM2);



    %% Backslash
    I=speye(n1);
    X=zeros(n1,n2);
    tt=tic;
    for i=1:n2
        X(:,i)=(A+s(i)*I)\C;
    end
    time_backslash=toc(tt);
    disp('********')
    fprintf("Backslash - Execution time: %e\n", time_backslash);
end
% Heatmaps (Figure 2)
subplot(1,2,1)
imagesc(log(abs(RES_tot)))
x1=xlabel('Iterations');
set(x1,'interpreter','latex')
y1=ylabel('Shift index');
set(y1,'interpreter','latex')
title('Our poles')
colorbar
subplot(1,2,2)
imagesc(log(abs(RES2)))
x1=xlabel('Iterations');
set(x1,'interpreter','latex')
y1=ylabel('Shift index');
set(y1,'interpreter','latex')
colorbar
title('ADM poles')
