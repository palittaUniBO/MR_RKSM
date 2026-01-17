% Example 2.2 in
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

close all
clear all
addpath(genpath('../src'))
warning off

n=200;
rng('default')
Q=randn(n);
kappa=norm(Q,'fro')*norm(inv(Q),'fro');
A=Q*diag([-100:-1,1:100])/Q;
l=300;

b=randn(n,1);
b=b/norm(b);
t=1;

for k=3:7

    s1=.5+1i*randn(l,1)*10^(-k);
    s2=1-1e-10+1i*randn(l,1)*10^(-k);
    

    X1=zeros(n,l);
    X2=X1;
    for i=1:l
        X1(:,i)=(A+s1(i)*speye(n))\b;
        X2(:,i)=(A+s2(i)*speye(n))\b;
    end

    ss1=svd(X1);
    ss2=svd(X2);
    ratio1(t)=ss1(2)/ss1(1);
    ratio2(t)=ss2(2)/ss2(1);

    XX1=((A+mean(s1)*speye(n))\b)*ones(1,l);
    XX2=((A+mean(s2)*speye(n))\b)*ones(1,l);
    
    err1(t)=norm(X1-XX1,'fro')/norm(X1,'fro');
    err2(t)=norm(X2-XX2,'fro')/norm(X2,'fro');
    
    t=t+1;

end

T=table((3:7)',err1',ratio1',err2',ratio2');
T.Properties.VariableNames={'Clustering Level k','Error 1','Ratio Singular Values 1',...
    'Error 2','Ratio Singular Values 2'}