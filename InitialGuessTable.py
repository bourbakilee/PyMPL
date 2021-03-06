# 2015.07.30, LI Yunsheng
# Initial Guess Table for spiral calculation

# sqlite3 dataase
import sqlite3
import TrajectoryGeneration as TG
import numpy as np
from numpy import matlib

def db_create():
    # executed only one time
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()
    cursor.execute('''create table InitialGuessTable(
                    k0 int,
                    x1 int,
                    y1 int,
                    theta1 int,
                    k1 int,
                    p1 double,
                    p2 double,
                    sg double,
                    primary key(k0,x1,y1,theta1,k1));''')
    simple_data = [(0,i,0,0,0,0.,0.,1.+i*49/16) for i in range(17)]
    cursor.executemany('insert or ignore into InitialGuessTable values (?,?,?,?,?,?,?,?)',simple_data)
    conn.commit()
    cursor.execute('select * from InitialGuessTable;')
    print(cursor.fetchall())
    cursor.close()
    conn.close()


def db_list_all(cursor):
    '''list all items in table'''
    cursor.execute('select * from InitialGuessTable;')
    print(cursor.fetchall())


def db_query(cursor, word):
    # word = (_k0, _x1, _y1, _theta1, _k1)
    cursor.execute('select p1, p2, sg  from InitialGuessTable where \
                               k0=? and x1=? and y1=? and theta1=? and k1=?;', word )
    return cursor.fetchone()


def neighbors(cursor, key):
    i,j,k,l,m = key
    ns = []
    if (i+1) <= 8:
        ns.append((i+1,j,k,l,m))
    if (i-1) >= -8:
        ns.append((i-1,j,k,l,m))
    if (j+1) <= 16:
        ns.append((i,j+1,k,l,m))
    if (j-1) >= 0:
        ns.append((i,j-1,k,l,m))
    if (k+1) <= 8:
        ns.append((i,j,k+1,l,m))
    if (k-1) >= -8:
        ns.append((i,j,k-1,l,m))
    if (l+1) <= 8:
        ns.append((i,j,k,l+1,m))
    if (l-1) >= -8:
        ns.append((i,j,k,l-1,m))
    if (m+1) <= 8:
        ns.append((i,j,k,l,m+1))
    if (m-1) >= -8:
        ns.append((i,j,k,l,m-1))
    return [n for n in ns if db_query(cursor, n) is None]


# this build algorithm is just like the Dijkstra Graph Search Algorithm~~~~
def Build_InitialGuessTable():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()
    key_queue = [(0,i,0,0,0) for i in range(17)] # current daten in database. can also direct select from database
    # print(key_queue)
    items = 17
    while len(key_queue)>0:
        key = key_queue.pop(0)
        # print(key)
        init_val = db_query(cursor, key)
        # print(init_val)
        nears = neighbors(cursor, key) # select keys whose Manhattan Distance to key equals to 1.
        # print(nears)
        for (i,j,k,l,m) in nears:
            bd_con = (i/40, 1+3.0625*j, 6.25*k, l*np.pi/16, m/40)
            val = TG.optimize(bd_con, init_val)
            key_val = (i,j,k,l,m, val[0], val[1], val[2])
            items += 1
            print('Progress: {0:%}, Content: {1}'.format(items/1419857, key_val))
            cursor.execute('insert or ignore into InitialGuessTable values (?,?,?,?,?,?,?,?)', key_val)
            key_queue.append((i,j,k,l,m))
        # 
        conn.commit()
    cursor.close()
    conn.close()


def DelInvalidItems():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()
    cursor.execute('delete from InitialGuessTable where sg<0')
    conn.commit()
    cursor.close()
    conn.close()


def test():
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()
    cursor.execute('select count(*) from InitialGuessTable where sg<0')
    fail = cursor.fetchone()[0]
    cursor.execute('select count(*) from InitialGuessTable')
    records = cursor.fetchone()[0]
    print(fail,records,records-fail,fail/records)
    cursor.close()
    conn.close()


def test_select():
    bd_con = (3.9/40, 7.1*3.0625+1, 2.1*6.25, 3.9*3.141592654/16, 4.1/40)
    conn = sqlite3.connect('InitialGuessTable.db')
    cursor = conn.cursor()
    initial_guess = TG.select_init_val(cursor, bd_con)
    print(initial_guess)

    initial_guess2 = (0.0145, 0.0037, 23.5967)
    pp = TG.optimize(bd_con, initial_guess2)
    print(pp)

if __name__ == '__main__':
    # db_create()
    # Build_InitialGuessTable()
    test()
    # DelInvalidItems()
