# Zagadnienie przeszukiwania

### Minimalizacja fukcji celu

### Dwie metody:
- metoda najszybszego spadku gradientu
- metody netwona

# TODO - Kod
- Rysowanie wartości kolorów
- Dodać warunek stopu epsilon

## Metody
### Steepest Gradient Descent
```
x <- x_0
while !stop:
    d <- grad_q(x)
    x <- x + B_t * d
```

### Newton Method
```
x <- x_0
while !stop:
    d <- inv_hess_q(x) * grad_q(x)
    x <- x + B_t * d
```

`inv_hess - odwrotność hesjanu`

`grad - gradient`


## Dodatkowe info
- punkty należy traktować jako wektory
`[x, y, z, ...]`
- losowanie osobno składowych gradientu nie jest poprawne !!!
