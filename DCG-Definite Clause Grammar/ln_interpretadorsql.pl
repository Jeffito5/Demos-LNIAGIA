:-encoding(utf8).

%tabela(IdTab,DescricaoLN).

%tabela('Vendedores', [vendedores]).

%atributo(IdTab,IdAtributo,DescricaoAtributoLN).

atrib('Vendedores','IdVend',[vendedores]).
atrib('Vendedores','NVendas',[número,de,vendas]).
atrib('Vendedores','Montante',[montante,de,vendas]).

% Frases
% Quais sao os vendedores cujo número de vendas é maior que 100?
% Quais sao os vendedores cujas vendas são superiores a 100?
% Quais sao os vendedores cujo número de vendas é superior a 100 e cujo
% montante de vendas é menor que 1000?

% Exemplos de utilizacao:
% ?- frase([quais,são,os,vendedores,cujo,número,de,vendas,é,maior,que,100],[]).
% ?- frase([quais,são,os,vendedores,cujas,vendas,são,superiores,a,100],[]).
% ?- frase([quais,são,os,vendedores,cujo,número,de,vendas,é,superior,a,100,e,cujo,montante,de,vendas,é,menor,que,1000],[]).

% Estruturas
% select(Atributo,IdTab,[(Atrib1,Cond,Val),...])

frase(L,L1):-frase_tipo1(Estrutura,L,L1),traduz(Estrutura,SQL),write(SQL).
% frase(L,L1):-frase_tipo2(Estrutura,L,L1),traduz(Estrutura,SQL),write(SQL).
frase(_,_):-write('Frase desconhecida').


frase_tipo1(Estrutura)-->frase_select(Estrutura).

frase_select(select(Atrib,IdTab,L))-->
	interrogacao(Num),
	complemento_directo(Num,(Atrib,IdTab)),
	sequencia_complemento_indirecto(L).

complemento_directo(Num,Atrib)-->
	artigo(Num,Gen),
	atributo(Num,Gen,Atrib).


sequencia_complemento_indirecto([Cond])-->
	complemento_indirecto(Cond).
sequencia_complemento_indirecto([Cond|T])-->
	complemento_indirecto(Cond),
	particula_conjuntiva,
	sequencia_complemento_indirecto(T).

complemento_indirecto((Atrib,Cond,Val))-->
	preposicao_relativa(Num,Gen),
	atributo(Num,Gen,(Atrib,_)),
	comparacao(Cond,Num),
	valor(Val).


comparacao('>',Num)-->
	verbo(Num),
	([maior,que];[superior,a]).
comparacao('<',Num)-->
	verbo(Num),
	([menor,que];[inferior,a]).
comparacao('>',Num)-->
	verbo(Num),
	([maiores,que];[superiores,a]).
comparacao('<',Num)-->
	verbo(Num),
	([menores,que];[inferiores,a]).


valor(Val)-->[Val],{integer(Val)}.


interrogacao(Num)--> pronome(Num),verbo(Num).
pronome(sin) --> [qual].
pronome(plu) --> [quais].
verbo(sin) --> [é].
verbo(plu) --> [são].
terminador-->[fim];[terminar];[sair];[fechar].
particula_conjuntiva-->[e].
artigo(sin,mas)-->[o].
artigo(sin,fem)-->[a].
artigo(plu,mas) --> [os].
artigo(plu,fem) --> [as].
preposicao_relativa(plu,mas)-->[cujos].
preposicao_relativa(sin,mas)-->[cujo].
preposicao_relativa(plu,fem)-->[cujas].
preposicao_relativa(sin,fem)-->[cuja].
atributo(plu,mas,(IdAtributo,IdTab))-->[vendedores],{atrib(IdTab,IdAtributo,[vendedores])}.
atributo(sin,mas,(IdAtributo,IdTab))-->[vendedor],{atrib(IdTab,IdAtributo,[vendedores])}.
atributo(sin,mas,(IdAtributo,IdTab))-->[número,de,vendas],{atrib(IdTab,IdAtributo,[número,de,vendas])}.
atributo(plu,fem,(IdAtributo,IdTab))-->[vendas],{atrib(IdTab,IdAtributo,[número,de,vendas])}.
atributo(sin,mas,(IdAtributo,IdTab))-->[montante,de,vendas],{atrib(IdTab,IdAtributo,[montante,de,vendas])}.

%Geracao da resposta SQL

traduz(select(Atrib,IdTab,L),SQL):-
    atomics_to_string(['SELECT', Atrib, 'FROM', IdTab, 'WHERE'], ' ', SQL1),mensagem(L,SQL1,SQL).



mensagem([(Atrib,Cond,Val)],SQL1,SQL):- !,
    atomics_to_string([SQL1,Atrib,Cond,Val], ' ',SQL).

mensagem([(Atrib,Cond,Val)|T],SQL1,SQL):-
    atomics_to_string([SQL1,Atrib,Cond,Val,' AND'], ' ',SQL2),mensagem(T,SQL2,SQL).










