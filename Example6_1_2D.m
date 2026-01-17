% Example 6.1: 2-D Convection-diffusion example in
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

n=100;
epsilon=0.5;
h = 1 / (n - 1);

T = -epsilon / h^2 *...
    spdiags([-ones(n,1), 2 * ones(n,1), -ones(n,1)], -1:1, n, n);
% negative first derivative
N = -1 / (2 * h) *...
    spdiags([-ones(n, 1), zeros(n, 1), ones(n, 1)], -1:1, n, n);

x = linspace(0, 1, n);
phi1=3*(1-x.^2);
psi1=x;
phi2=x;
psi2=-2*(1-x.^2);

PHI1=spdiags(phi1',0,n,n);    
PSI1=spdiags(psi1',0,n,n);    
PHI2=spdiags(phi2',0,n,n);    
PSI2=spdiags(psi2',0,n,n);   
I=speye(n);

B1=PHI1*N;
B2= PSI2* N;

A =kron(I,T)+kron(T,I)+kron(B2,PHI2)+kron(PSI1,B1');


% number of shifts
n2 = 1000;

% right-hand side
k = 1; % rank
rng('default')
C = randn(n^2, k);
C=C/norm(C);
D = kron(ones(n2, 1),eye(k)); 
normb = sqrt(abs(diag(D*(C'*C)*D')));

% maximum number of iterations
iter = 100;

% convergence tolerance
tol = 1e-8;

for p=1:3

    if p==1
        % real shifts
        s = -logspace(-6,6,n2);
        fprintf('********\n REAL SHIFTS\n********\n')
    elseif p==2
        % complex shifts coming in conjugate pairs 
        s = -logspace(-6,6,n2/2);
        s = [1i*s, -1i*s];
        
        fprintf('********\n COMPLEX SHIFTS (CONJ PAIRS)\n********\n')

    else
        % complex shifts not coming in conjugate pairs
        center = -2*1.190372229367327e+01 + 5*1i -200;
        radius = 500;
        vscale = 1;
        s = zeros(n2,1);
        for i = 1 : n2
            theta = 2.0*pi*i/n2;
            s(i) = center + radius* (cos(theta)+1i*vscale*sin(theta));
        end
        fprintf('********\n COMPLEX SHIFTS (NON CONJ PAIRS)\n********\n')
    end

    % the matrix with shifts
    B = diag(sparse(s));
    
    
    %% Rational Krylov

    params.m=iter;
    params.tol=tol;
    params.shift_init=s(1);
    params.ch=0;

    tt=tic;
    [U,V,it,res,shifts] = rational_ksm_shifted_systems(A,s,C,params);
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


    %% Extended Krylov 
    tt=tic;
    [U,V,er2]=EKSM_ShiftedLS(A,s,C,D,params);
    time_EKSM = toc(tt);

    Z=U*V;
    RES=A*Z+Z*BB-C*D';
    normr=zeros(n2,1);
    for t=1:n2*k
        normr(t)=norm(RES(:,t));
    end  
    disp('********')
    fprintf("Extended Krylov\n max (column-wise) 2-norm of relative residuals: %e\n",max(normr./normb));
    fprintf("Iterations: %d, Solution rank: %d, Execution time: %e\n", size(U,2)/2,size(U,2),time_EKSM);

    %% Restarted shifted FOM

    m=100;
    kmax=10;

    tt=tic;
    [Z,it,cycle,res_FOM] = restarted_shiftedFOM(A,s,C,tol,m,kmax);
    time_FOM=toc(tt);

    RES=A*Z+Z*BB-C*D';
    normr=zeros(n2,1);
    for t=1:n2*k
    normr(t)=norm(RES(:,t));
    end  
    disp('********')
    fprintf("FOM\n max (column-wise) 2-norm of relative residuals: %e\n",max(normr./normb));
    fprintf("Execution time: %e, Total iterations: %d, cycles: %d\n", time_FOM, (cycle-1)*m+it,cycle);
end

% plotting residual in case of complex shifts not coming in conjugate pairs
% (Figure 1)
semilogy(res,'LineWidth',2)
hold on
semilogy(er2,'r--','LineWidth',2)
semilogy(res_FOM(1:100),'k:','LineWidth',2)
leg=legend('MR-RKSM','G-EKSM','FOM(100)');
set(leg,'interpreter','latex')