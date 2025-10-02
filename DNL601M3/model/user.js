const mongoose =require('mongoose')
const Schema=mongoose.Schema

const schema = mongoose.Schema(
    {
        name: String,
        email:String,
        phone:String,
        phone2:String,
        password: String,
        access:Boolean,
        orders: [{type:Schema.Types.ObjectId,ref:"Order"}]
    }
);
module.exports=mongoose.model("User",schema);