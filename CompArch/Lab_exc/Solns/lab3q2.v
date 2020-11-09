module seq(clk,rst,in,out);
input clk,in,rst;
output out;
reg out;
reg[2:0] state;
always@(posedge clk,posedge rst)
begin
if(rst)begin
	state <= 3'b000;
	out <= 0;
end
	else begin
	case(state)
	3'b000: if(in) begin
		state <= 3'b001;
		out <= 0;
	end
	else begin
		state <= 3'b000;
		out <= 0;
	end
	3'b001: if(in) begin
		state <= 3'b001;
		out <= 0;
	end
	else begin
		state <= 3'b010;
		out <= 0;
	end
	3'b010: if(in) begin
		state <= 3'b011;
		out <= 0;
	end
	else begin
		state <= 3'b000;
		out <= 0;
	end
	3'b011: if(in) begin
		state <= 3'b100;
		out <= 0;
	end
	else begin
		state <= 3'b010;
		out <= 0;
	end
	3'b100: if(in) begin
		state <= 3'b001;
		out <= 0;
	end
	else begin
		state <= 3'b010;
		out <= 1;
	end
	default: begin //don't forget default
		state <= 3'b000;
		out <= 0;
	end
	endcase
	end
end
endmodule

module test;
reg clk,in,rst;
reg[15:0] seq;
wire out;
integer i;
seq a(clk,rst,in,out);
initial
begin
$monitor($time," Reset = %b STATE=%3b",rst,a.state);
clk=0;
rst=1; // X -> 1 is a posedge!! See how this is used to set initial state
seq = 16'b1011001011010001;
#5 rst=0;
for(i=0;i<16;i=i+1) begin
	in = seq[i];
	#2 clk=1;
	#2 clk=0;
	$display($time," STATE=%3b SEQ=%b out=%b",a.state,seq[i],out); // see how state is accessed and retention of values
end
end
endmodule