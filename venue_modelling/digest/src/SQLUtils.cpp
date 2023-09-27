#include "SQLUtils.h"

/// @brief return int from sql server using input sql string
/// @param sql 
/// @param valueName 
/// @return int
int selectInt(std::string sql, std::string valueName){

    int returnVal;

    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "webuser", "webuser");
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
        std::cout << "# ERR: SQLException in " << __FILE__;
        std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
        std::cout << "# ERR: " << e.what();
        std::cout << " (MySQL error code: " << e.getErrorCode();
        std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
    }
    
    return returnVal;
}

/// @brief return float from sql server using input sql string
/// @param sql 
/// @param valueName 
/// @return float
float selectFloat(std::string sql, std::string valueName){

    float returnVal;

    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "webuser", "webuser");
        /* Connect to the MySQL test database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery(sql);

        while (res -> next()) {
            /* Access column data by alias or column name */
            returnVal = (float)(res -> getDouble(valueName)); //cast double result to float as cppconn does not have a getFloat()
        }

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        std::cout << "# ERR: SQLException in " << __FILE__;
        std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
        std::cout << "# ERR: " << e.what();
        std::cout << " (MySQL error code: " << e.getErrorCode();
        std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
    }
    
    return returnVal;
}

/// @brief return std::string from sql server using input sql string
/// @param sql 
/// @param valueName 
/// @return std::string
std::string selectString(std::string sql, std::string valueName){

    std::string returnVal;

    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "webuser", "webuser");
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
        std::cout << "# ERR: SQLException in " << __FILE__;
        std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
        std::cout << "# ERR: " << e.what();
        std::cout << " (MySQL error code: " << e.getErrorCode();
        std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
    }
    
    return returnVal;
}

/// @brief set ticket status to ready for username
/// @param username 
void ticketReady(std::string username){
    //sql to set ready on ticketTable to 1 for username
    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "webuser", "webuser");
        /* Connect to the MySQL database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery("UPDATE ticketTable SET ready = 1 WHERE username = '" + username + "'");

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        
        //unholy behaivour :O - this always gives err 0 as no data gets returned so this is what it is
        if (e.getErrorCode() != 0){
            std::cout << "# ERR: SQLException in " << __FILE__;
            std::cout << " (" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
            std::cout << "# ERR: " << e.what();
            std::cout << " (MySQL error code: " << e.getErrorCode();
            std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
        }
    }
}