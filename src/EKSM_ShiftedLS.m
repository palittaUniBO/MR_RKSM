function [V,Y,er2]=EKSM_ShiftedLS(A,Bdiag,C,D,params)
% function [V,Y,er2]=EKSM_ShiftedLS(A,Bdiag,C,D,params)
%
% Approximately solve
%
%       A X  + X Bdiag = C*D'
%
% by means of the extended Krylov subspace method
%
%
%
% Input
%  A   coeff matrix
%  Bdiag   diagonal matrix containing shifts
%  C, D    low-rank factors of the rhs
%  params   computational parameters
%        ---> params.m   max space dimension
%        ---> params.tol stopping tolerance on the column-wise relative
%        residual norm
%        ---> params.iterative  0: direct solution of the linear systems
%                                  with A (default parameter)
%                               1: iterative solution of the linear systems
%                                  with A by gmres(50) preconditioned by ILU
%
%  Output:
%  V, Y   solution low-rank factor   X = V*Y
%  er2 history of scaled residual, as above
%
%
% 
% This code has been designed by modyfing Valeria Simoncini's solver kpik
% for Lyapunov equations, see
%
% V. Simoncini
% A new iterative method for solving large-scale Lyapunov matrix equations, 
% SIAM J.  Scient. Computing, v.29, n.3 (2007), pp. 1268-1288.
%
%
%THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
%IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
%FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
%COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
%IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
%CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 


rhs=C;
m=params.m;
tol=params.tol;
nrm_rhs=sqrt(diag(D*(C'*C)*D'));
ll=length(Bdiag);


if ~isfield(params,'iterative') || ~params.iterative
    params.iterative=0;
    [LA,UA]=lu(A);
else
    [LL,UU]=ilu(A);
end
        

k_max = m;

n=size(A,1);
sh=size(C,2);
s=2*sh;
if ~params.iterative
    rhs1=UA\(LA\rhs);
else
    rhs1=gmres(A,rhs,50,tol/10,100,LL,UU);
end
[U(1:n,1:s),beta]=qr([rhs,rhs1],0);
%ibeta=inv(beta(1:s,1:s));
beta = beta(1:sh,1:sh); 
Cm= eye(s*(m+2),sh)*beta;
Y=zeros((m+1)*s,ll);

H=zeros((m+1)*s,m*s);
T=zeros((m+1)*s,m*s);
%L=zeros((m+1)*s,m*s);
%odds=[];
er2=zeros(m+1,1);


for j=1:m

    % basis expansion
    jms=(j-1)*s+1;j1s=(j+1)*s;js=j*s;js1=js+1; jsh=(j-1)*s+sh;
    Up(1:n,1:sh) = A*U(:,jms:jsh); 
    if params.iterative
        Up(1:n,sh+1:s)=gmres(A,U(1:n,jsh+1:js),50,tol/10,100,LL,UU);
    else
        Up(1:n,sh+1:s) = UA\(LA\U(1:n,jsh+1:js)); 
    end 
    
    
    %new bases block (modified gram)
    for l=1:2
        k_min=max(1,j-k_max);
        for kk=k_min:j
            k1=(kk-1)*s+1; k2=kk*s;
            coef= U(1:n,k1:k2)'*Up;
            H(k1:k2,jms:js) = H(k1:k2,jms:js)+ coef;
            Up = Up - U(:,k1:k2)*coef; 
        end
    end

  if (j<=m)
    [Up,H(js1:j1s,jms:js)] = qr(Up,0);
    hinv=inv(H(js1:j1s,jms:js));
  end

  %% RECURSIVE COMPUTATION OF T=V'*A*V. WE COMPUTE AN EXPLICIT PROJECTION INSTEAD FOR STABILITY PURPOSES
  % if (j==1)
  %     L(1:j*s+sh,(j-1)*sh+1:j*sh) =...
  %     [ H(1:s+sh,1:sh)/ibeta(1:sh,1:sh), speye(s+sh,sh)/ibeta(1:sh,1:sh)]*ibeta(1:s,sh+1:s);
  % else
  %     L(1:j*s+s,(j-1)*sh+1:j*sh) = L(1:j*s+s,(j-1)*sh+1:j*sh) + H(1:j*s+s,jms:jms-1+sh)*rho;
  % end
  % 
  % odds = [odds, jms:(jms-1+sh)];   % store the odd block columns
  % evens = 1:js; evens(odds)=[];
  % T(1:js+s,odds)=H(1:js+s,odds);   %odd columns
  % 
  % T(1:js+sh,evens)=L(1:js+sh,1:j*sh);   %even columns
  % L(1:j*s+s,j*sh+1:(j+1)*sh) = ...
  %      ( I(1:j*s+s,(js-sh+1):js)- T(1:js+s,1:js)*H(1:js,js-sh+1:js))*hinv(sh+1:s,sh+1:s);
  % rho = hinv(1:sh,1:sh)\hinv(1:sh,sh+1:s);
  
  % compute explicit projection for stability
  T(1:j1s,1:j1s)=[U(:,1:js),Up]'*A*[U(:,1:js),Up];
  I=eye(js+s);
  
  % solve the projected shifted linear systems
  for t=1:ll
      Y(1:js,t)=(T(1:js,1:js)+Bdiag(t)*I(1:js,1:js))\(Cm(1:js,:)*D(t,:)');
  end 
  % compute residuals
  cc=T(js1:j1s,js-s+1:js);
  RES=cc*Y(js-s+1:js,:);
  er2(j)=max(sqrt(diag(RES'*RES))./nrm_rhs);

  if (er2(j)<tol) 
     break
  else
     U(1:n,js1:j1s)=Up;
  end
end

V=U(1:n,1:js);
Y=Y(1:js,:);
er2=er2(1:j);

return
