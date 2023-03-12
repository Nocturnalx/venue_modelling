#include <iostream>
#include <string.h>
#include <fstream>
#include <chrono>
#include <thread>

#include "mysql_connection.h"

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>

using namespace std;
using namespace std::this_thread; // sleep_for, sleep_until
using namespace std::chrono; // nanoseconds, system_clock, seconds

string username;

int selectInt(string sql, string valueName){

    int returnVal;

    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "root");
        /* Connect to the MySQL test database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery(sql);

        while (res -> next()) {
            /* Access column data by alias or column name */
            returnVal = res -> getInt(valueName);
        }

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
    
    return returnVal;
}

string selectString(string sql, string valueName){

    string returnVal;

    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "root");
        /* Connect to the MySQL test database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery(sql);

        while (res -> next()) {
            /* Access column data by alias or column name */
            returnVal = res -> getString(valueName);
        }

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
    
    return returnVal;
}

void ticketReady(string username){
    //sql to set ready on ticketTable to 1 for username
    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "root");
        /* Connect to the MySQL test database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery("UPDATE ticketTable SET ready = 1 WHERE username = '" + username + "'");

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
}

int main(void){
    while(true){
        int cnt = selectInt("SELECT COUNT(username) AS cnt FROM ticketTable WHERE ready = 0", "cnt");

        if (cnt > 0){
            username = selectString("SELECT username FROM ticketTable WHERE ready = 0 LIMIT 1", "username");
            
            cout << "converting " << username << endl;

            //create new file
            ofstream newFile("out/" + username);
            newFile << "This is a test file.";
            newFile << "this file was made for the account" + username;
            newFile.close();

            //delete input file
            string path = "in/" + username;
            unlink(path.c_str());

            //set ticket to ready
            ticketReady(username);
        }

        sleep_for(seconds(5));
    }

    return EXIT_SUCCESS;
}